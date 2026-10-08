"""LIBERO rollout of the frozen, official pi0.5 checkpoint without Mone."""

import argparse
from collections import deque
import json
from pathlib import Path

import numpy as np
import torch

from eval_mone_heldout import observation
from eval_mone_rollout import SUITE_LIMITS, WAIT_ACTION, sim_observation
from train_mone_adapter_contrastive import make_policy
from openpi import transforms
from openpi.policies import libero_policy
from openpi.shared import normalize
from openpi.training import config


@torch.no_grad()
def run_trial(args, env, initial_state, description, task_id, trial, model, policy,
              output_transform, device):
    env.seed(args.seed + trial)
    env.reset()
    obs = env.set_init_state(initial_state)
    for _ in range(args.wait_steps):
        obs, _, _, _ = env.step(WAIT_ACTION)
    initial_eef = np.asarray(obs['robot0_eef_pos'], dtype=np.float64)
    plan = deque()
    action_min = np.full(7, np.inf)
    action_max = np.full(7, -np.inf)
    action_abs_sum = np.zeros(7)
    replans = 0
    success = False
    for step in range(SUITE_LIMITS[args.suite]):
        if not plan:
            current_raw = sim_observation(obs, description, args.resize_size)
            current, _ = observation(policy._input_transform, current_raw, device)
            generator = torch.Generator(device=device).manual_seed(
                args.seed + task_id*100000 + trial*1000 + replans)
            noise = torch.randn((1, model.config.action_horizon, model.config.action_dim),
                                generator=generator, device=device, dtype=torch.float32)
            actions = model.sample_actions(device, current, noise=noise, num_steps=args.num_steps)
            result = output_transform(dict(state=current.state[0].detach().cpu().numpy(),
                                           actions=actions[0].detach().float().cpu().numpy()))
            chunk = np.asarray(result['actions'], dtype=np.float64)
            if chunk.shape != (model.config.action_horizon, 7) or not np.isfinite(chunk).all():
                raise ValueError(f'Invalid LIBERO actions: {chunk.shape}')
            plan.extend(chunk[:args.replan_steps])
            replans += 1
        action = plan.popleft()
        action_min = np.minimum(action_min, action)
        action_max = np.maximum(action_max, action)
        action_abs_sum += np.abs(action)
        obs, _, done, _ = env.step(action.tolist())
        if done:
            success = True
            break
    steps = step + 1
    return dict(suite=args.suite, task_id=task_id, task=description, trial=trial,
                success=success, steps=steps, replans=replans,
                action_min=action_min.tolist(), action_max=action_max.tolist(),
                action_abs_mean=(action_abs_sum / steps).tolist(),
                eef_displacement=float(np.linalg.norm(np.asarray(obs['robot0_eef_pos'])-initial_eef)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--suite', choices=tuple(SUITE_LIMITS), required=True)
    parser.add_argument('--task-id', type=int, required=True)
    parser.add_argument('--trials', type=int, default=5)
    parser.add_argument('--wait-steps', type=int, default=10)
    parser.add_argument('--replan-steps', type=int, default=5)
    parser.add_argument('--num-steps', type=int, default=10)
    parser.add_argument('--resize-size', type=int, default=224)
    parser.add_argument('--seed', type=int, default=7)
    args = parser.parse_args()
    if min(args.trials, args.wait_steps, args.replan_steps, args.num_steps) < 1:
        parser.error('Counts must be positive')
    if args.replan_steps > 10:
        parser.error('Replan interval exceeds action horizon')
    if args.output.exists() or args.output.with_suffix('.jsonl').exists():
        raise FileExistsError(args.output)
    if not torch.cuda.is_available():
        raise RuntimeError('A CUDA GPU is required for LIBERO rollout')
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    suite = benchmark.get_benchmark_dict()[args.suite]()
    if not 0 <= args.task_id < suite.n_tasks:
        parser.error(f'Invalid task id {args.task_id} for {args.suite}')
    task = suite.get_task(args.task_id)
    device = torch.device('cuda:0')
    model, policy = make_policy(args.checkpoint, device)
    norm_stats = normalize.load(args.checkpoint / 'assets/physical-intelligence/libero')
    cfg = config.get_config('pi05_libero').model
    output_transform = transforms.compose([
        *config.ModelTransformFactory()(cfg).outputs,
        transforms.Unnormalize({'actions': norm_stats['actions']}, use_quantiles=True),
        libero_policy.LiberoOutputs(),
    ])
    init_path = Path(get_libero_path('init_states')) / task.problem_folder / task.init_states_file
    initial_states = torch.load(init_path, weights_only=False)
    if len(initial_states) < args.trials:
        raise ValueError(f'Only {len(initial_states)} official initial states')
    bddl = Path(get_libero_path('bddl_files')) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    try:
        with args.output.with_suffix('.jsonl').open('x') as stream:
            for trial in range(args.trials):
                row = run_trial(args, env, initial_states[trial], task.language, args.task_id,
                                trial, model, policy, output_transform, device)
                rows.append(row)
                stream.write(json.dumps(row) + '\n')
                stream.flush()
                print(f"[baseline] {args.suite}:{args.task_id} trial={trial} "
                      f"success={row['success']} steps={row['steps']}", flush=True)
    finally:
        env.close()
    result = dict(status='complete', checkpoint=str(args.checkpoint.resolve()),
                  suite=args.suite, task_id=args.task_id, task=task.language,
                  trials=args.trials, successes=sum(row['success'] for row in rows),
                  success_rate=sum(row['success'] for row in rows)/len(rows),
                  note='Small baseline sanity check, not benchmark-wide success estimate.')
    args.output.write_text(json.dumps(result, indent=2))
    print(f"[done] {args.output} success={result['successes']}/{result['trials']}")


if __name__ == '__main__':
    main()
