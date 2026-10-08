"""Paired LIBERO success on held-out tasks for 18-layer Mone and controls."""

import argparse
from collections import deque
import json
from pathlib import Path

import numpy as np
import torch

if __package__:
    from .eval_mone_heldout import observation, rows_for
    from .eval_mone_rollout import SUITE_LIMITS, WAIT_ACTION, sim_observation
    from .eval_mone_all_layers import file_sha256, load_adapter
    from .plan_mone_full import digest
    from .train_mone_adapter_contrastive import make_policy, raw_observation
else:
    from eval_mone_heldout import observation, rows_for
    from eval_mone_rollout import SUITE_LIMITS, WAIT_ACTION, sim_observation
    from eval_mone_all_layers import file_sha256, load_adapter
    from plan_mone_full import digest
    from train_mone_adapter_contrastive import make_policy, raw_observation
from openpi import transforms
from openpi.models_pytorch.mone_all_layers import capture_context_layers
from openpi.policies import libero_policy
from openpi.shared import normalize
from openpi.training import config


MODES = ('baseline', 'relevant', 'same_task', 'fixed_history', 'parameter_control')


@torch.no_grad()
def trial(args, env, initial_state, description, task_id, initial_index, mode,
          donor_rows, fixed_features, fixed_mask, model, policy, memory,
          control, output_transform, device):
    env.seed(args.seed + initial_index)
    env.reset()
    obs = env.set_init_state(initial_state)
    history = deque(maxlen=11)
    history.append(sim_observation(obs, description, args.resize_size))
    for _ in range(args.wait_steps):
        obs, _, _, _ = env.step(WAIT_ACTION)
        history.append(sim_observation(obs, description, args.resize_size))
    static_features = static_mask = None
    if mode == 'static_initial':
        initial, _ = observation(policy._input_transform, history[0], device)
        static_features, static_mask = capture_context_layers(
            model, initial, args.history_tokens)
    plan = deque()
    replans = 0
    success = False
    for step in range(SUITE_LIMITS[args.suite]):
        if not plan:
            current, _ = observation(policy._input_transform, history[-1], device)
            adapter = memory
            features = mask = None
            if mode == 'relevant':
                past, _ = observation(policy._input_transform, history[0], device)
                features, mask = capture_context_layers(model, past, args.history_tokens)
            elif mode == 'same_task':
                donor_frame = min(len(donor_rows)-1,
                                  round(step/max(1, SUITE_LIMITS[args.suite]-1)*(len(donor_rows)-1)))
                donor, _ = observation(policy._input_transform,
                                       raw_observation(donor_rows[donor_frame], description), device)
                features, mask = capture_context_layers(model, donor, args.history_tokens)
            elif mode == 'static_initial':
                features, mask = static_features, static_mask
            elif mode == 'current_frame':
                features, mask = capture_context_layers(model, current, args.history_tokens)
            elif mode == 'fixed_history':
                features, mask = fixed_features, fixed_mask
            elif mode == 'parameter_control':
                features, mask = capture_context_layers(model, current, args.history_tokens)
                adapter = control
            generator = torch.Generator(device=device).manual_seed(
                args.seed + task_id*100000 + initial_index*1000 + replans)
            noise = torch.randn((1, model.config.action_horizon, model.config.action_dim),
                                generator=generator, device=device, dtype=torch.float32)
            if mode == 'baseline':
                actions = model.sample_actions(device, current, noise=noise,
                                               num_steps=args.num_steps)
            else:
                states = adapter.build_states(features, mask)
                with adapter.activate(model, states):
                    actions = model.sample_actions(device, current, noise=noise,
                                                   num_steps=args.num_steps)
            outputs = output_transform(dict(state=current.state[0].detach().cpu().numpy(),
                                            actions=actions[0].detach().float().cpu().numpy()))
            chunk = np.asarray(outputs['actions'], dtype=np.float64)
            if chunk.shape != (model.config.action_horizon, 7) or not np.isfinite(chunk).all():
                raise ValueError(f'Invalid LIBERO action chunk: {chunk.shape}')
            plan.extend(chunk[:args.replan_steps])
            replans += 1
        obs, _, done, _ = env.step(plan.popleft().tolist())
        history.append(sim_observation(obs, description, args.resize_size))
        if done:
            success = True
            break
    return dict(suite=args.suite, task_id=task_id, task=description,
                trial=initial_index, mode=mode, success=success,
                steps=step+1, replans=replans)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--memory-adapter', type=Path, required=True)
    parser.add_argument('--control-adapter', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--suite', choices=tuple(SUITE_LIMITS), required=True)
    parser.add_argument('--trials', type=int, default=10)
    parser.add_argument('--history-tokens', type=int, default=32)
    parser.add_argument('--wait-steps', type=int, default=10)
    parser.add_argument('--replan-steps', type=int, default=5)
    parser.add_argument('--num-steps', type=int, default=10)
    parser.add_argument('--resize-size', type=int, default=224)
    parser.add_argument('--seed', type=int, default=7)
    args = parser.parse_args()
    if min(args.trials, args.history_tokens, args.wait_steps,
           args.replan_steps, args.num_steps) < 1 or args.replan_steps > 10:
        parser.error('Invalid rollout settings')
    if args.output.exists() or args.output.with_suffix('.jsonl').exists():
        raise FileExistsError(args.output)
    if not torch.cuda.is_available():
        raise RuntimeError('Rollout requires a CUDA GPU')
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    plan = json.loads(args.plan.read_text())
    plan_hash = plan.pop('plan_sha256')
    if digest(plan) != plan_hash:
        raise ValueError('Plan hash mismatch')
    selected = sorted(task for task in plan['partitions']['test']
                      if plan['task_lookup'][task]['suite'] == args.suite)
    if len(selected) != 2:
        raise ValueError(f'Expected two test tasks in {args.suite}, found {len(selected)}')
    device = torch.device('cuda:0')
    model, policy = make_policy(args.checkpoint, device)
    memory_meta = torch.load(args.memory_adapter, map_location='cpu', weights_only=False)
    backbone_hash = memory_meta['backbone_sha256']
    if file_sha256(args.checkpoint/'model.safetensors') != backbone_hash:
        raise ValueError('Rollout backbone differs from training checkpoint')
    memory, memory_saved = load_adapter(args.memory_adapter, plan_hash, backbone_hash, device)
    control, control_saved = load_adapter(args.control_adapter, plan_hash, backbone_hash, device)
    if (memory_saved['history_mode'], control_saved['history_mode']) != (
        'true_history', 'current_frame'):
        raise ValueError('Expected true-history and current-frame controls')
    if memory_saved['step'] != control_saved['step'] or memory_saved['seed'] != control_saved['seed']:
        raise ValueError('Controls differ in training steps or seed')
    if memory_saved['history_tokens'] != args.history_tokens or control_saved['history_tokens'] != args.history_tokens:
        raise ValueError('Rollout history token count differs from training')
    cfg = config.get_config('pi05_libero').model
    norm_stats = normalize.load(args.checkpoint/'assets/physical-intelligence/libero')
    output_transform = transforms.compose([
        *config.ModelTransformFactory()(cfg).outputs,
        transforms.Unnormalize({'actions': norm_stats['actions']}, use_quantiles=True),
        libero_policy.LiberoOutputs(),
    ])
    fixed_task = sorted(plan['partitions']['train'])[0]
    fixed_episode = plan['episodes'][fixed_task][0]['episode']
    fixed_rows = rows_for(args.data_dir, fixed_episode)
    fixed_obs, _ = observation(policy._input_transform,
                               raw_observation(fixed_rows[10], fixed_task), device)
    fixed_features, fixed_mask = capture_context_layers(model, fixed_obs, args.history_tokens)
    suite = benchmark.get_benchmark_dict()[args.suite]()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    records = []
    with args.output.with_suffix('.jsonl').open('x') as stream:
        for language in selected:
            task_id = plan['task_lookup'][language]['task_id']
            task = suite.get_task(task_id)
            if task.language != language:
                raise ValueError(f'Benchmark task changed: {language} vs {task.language}')
            donor_episode = plan['episodes'][language][2]['episode']
            donor_rows = rows_for(args.data_dir, donor_episode)
            init_path = Path(get_libero_path('init_states'))/task.problem_folder/task.init_states_file
            init_states = torch.load(init_path, weights_only=False)
            if len(init_states) < args.trials:
                raise ValueError(f'Only {len(init_states)} initial states for {language}')
            bddl = Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file
            env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
            try:
                for initial_index in range(args.trials):
                    for mode in MODES:
                        row = trial(args, env, init_states[initial_index], language,
                                    task_id, initial_index, mode, donor_rows,
                                    fixed_features, fixed_mask, model, policy, memory,
                                    control, output_transform, device)
                        records.append(row)
                        stream.write(json.dumps(row)+'\n')
                        stream.flush()
                        print(f"[rollout] {args.suite}:{task_id} trial={initial_index} "
                              f"mode={mode} success={row['success']} steps={row['steps']}", flush=True)
            finally:
                env.close()
    args.output.write_text(json.dumps(dict(status='complete', suite=args.suite,
                                           tasks=selected, trials=args.trials,
                                           checkpoint=str(args.checkpoint.resolve()),
                                           modes=MODES, records=len(records),
                                           plan_sha256=plan_hash,
                                           backbone_sha256=backbone_hash,
                                           memory_adapter=str(args.memory_adapter),
                                           control_adapter=str(args.control_adapter)), indent=2))
    print(f'[done] {args.output} records={len(records)}', flush=True)


if __name__ == '__main__':
    main()
