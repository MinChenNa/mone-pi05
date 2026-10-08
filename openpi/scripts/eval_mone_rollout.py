"""Paired LIBERO success-rate pilot using the trained L4 hook during sampling."""

import argparse
from collections import deque
import json
import math
from pathlib import Path

import numpy as np
import torch

if __package__:
    from .eval_mone_heldout import observation, rows_for
    from .train_mone_adapter_contrastive import make_policy, raw_observation
else:
    from eval_mone_heldout import observation, rows_for
    from train_mone_adapter_contrastive import make_policy, raw_observation
from openpi import transforms
from openpi.models_pytorch.mone_layer import LayerMemory, capture_history
from openpi.policies import libero_policy
from openpi.shared import normalize
from openpi.training import config
from openpi_client import image_tools


MODES = ('baseline', 'relevant', 'fixed_history')
SUITE_LIMITS = dict(libero_spatial=220, libero_object=280, libero_goal=300, libero_10=520)
WAIT_ACTION = [0.0] * 6 + [-1.0]


def quat_to_axisangle(quat):
    quat = np.asarray(quat, dtype=np.float64).copy()
    quat[3] = np.clip(quat[3], -1., 1.)
    denominator = math.sqrt(max(0., 1. - quat[3] ** 2))
    if math.isclose(denominator, 0.):
        return np.zeros(3, np.float32)
    return (quat[:3] * (2. * math.acos(quat[3]) / denominator)).astype(np.float32)


def sim_observation(obs, prompt, resize_size):
    base = np.ascontiguousarray(obs['agentview_image'][::-1, ::-1])
    wrist = np.ascontiguousarray(obs['robot0_eye_in_hand_image'][::-1, ::-1])
    base = image_tools.convert_to_uint8(image_tools.resize_with_pad(base, resize_size, resize_size))
    wrist = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist, resize_size, resize_size))
    state = np.concatenate((obs['robot0_eef_pos'], quat_to_axisangle(obs['robot0_eef_quat']),
                            obs['robot0_gripper_qpos'])).astype(np.float32)
    return {'observation/image': base, 'observation/wrist_image': wrist,
            'observation/state': state, 'prompt': str(prompt)}


def select_tasks(benchmark, allowed, suites, max_tasks):
    chosen = []
    for suite_name in suites:
        suite = benchmark.get_benchmark_dict()[suite_name]()
        for task_id in range(suite.n_tasks):
            task = suite.get_task(task_id)
            if task.language in allowed:
                chosen.append((suite_name, suite, task_id, task))
                break
        if len(chosen) == max_tasks:
            break
    if len(chosen) != max_tasks:
        raise ValueError(f'Found only {len(chosen)} of {max_tasks} requested held-out suite tasks')
    return chosen


def summarize(records):
    indexed = {}
    for row in records:
        key = (row['suite'], row['task_id'], row['trial'])
        indexed.setdefault(key, {})[row['mode']] = int(row['success'])
    if any(set(result) != set(MODES) for result in indexed.values()):
        raise ValueError('A paired trial is missing a control arm')
    rates = {mode: sum(result[mode] for result in indexed.values()) / len(indexed) for mode in MODES}
    paired = {}
    for control in ('baseline', 'fixed_history'):
        diffs = [result['relevant'] - result[control] for result in indexed.values()]
        paired[control] = dict(relevant_only=sum(x > 0 for x in diffs),
                               control_only=sum(x < 0 for x in diffs),
                               ties=sum(x == 0 for x in diffs), mean_difference=float(np.mean(diffs)))
    by_task = {}
    for suite, task_id, _ in indexed:
        key = f'{suite}:{task_id}'
        if key in by_task:
            continue
        trials = [v for (s, i, _), v in indexed.items() if s == suite and i == task_id]
        by_task[key] = {mode: sum(v[mode] for v in trials)/len(trials) for mode in MODES}
    return dict(paired_trials=len(indexed), success_rate=rates, paired=paired, by_task=by_task,
                note='Small paired rollout screen; not a benchmark-wide success estimate.')


@torch.no_grad()
def run_trial(args, env, initial_state, task_description, suite_name, task_id, trial, mode,
              model, policy, adapter, layer, fixed_state, output_transform, device):
    env.seed(args.seed + trial)
    env.reset()
    obs = env.set_init_state(initial_state)
    history = deque(maxlen=11)
    history.append(sim_observation(obs, task_description, args.resize_size))
    for _ in range(args.wait_steps):
        obs, _, _, _ = env.step(WAIT_ACTION)
        history.append(sim_observation(obs, task_description, args.resize_size))
    plan = deque()
    success = False
    replans = 0
    for step in range(SUITE_LIMITS[suite_name]):
        if not plan:
            current, _ = observation(policy._input_transform, history[-1], device)
            state = None
            if mode == 'relevant':
                past, _ = observation(policy._input_transform, history[0], device)
                features, mask = capture_history(model, past, layer, args.history_tokens)
                state = adapter.build_state(features, mask)
            elif mode == 'fixed_history':
                state = fixed_state
            generator = torch.Generator(device=device).manual_seed(
                args.seed + task_id*100000 + trial*1000 + replans)
            noise = torch.randn((1, model.config.action_horizon, model.config.action_dim),
                                generator=generator, device=device, dtype=torch.float32)
            if state is None:
                actions = model.sample_actions(device, current, noise=noise, num_steps=args.num_steps)
            else:
                with adapter.activate(model, state, layer):
                    actions = model.sample_actions(device, current, noise=noise, num_steps=args.num_steps)
            outputs = output_transform(dict(state=current.state[0].detach().cpu().numpy(),
                                            actions=actions[0].detach().float().cpu().numpy()))
            action_chunk = np.asarray(outputs['actions'])
            if action_chunk.shape[-1] != 7 or len(action_chunk) < args.replan_steps:
                raise ValueError(f'Invalid LIBERO action chunk: {action_chunk.shape}')
            plan.extend(action_chunk[:args.replan_steps])
            replans += 1
        obs, _, done, _ = env.step(plan.popleft().tolist())
        history.append(sim_observation(obs, task_description, args.resize_size))
        if done:
            success = True
            break
    return dict(suite=suite_name, task_id=task_id, task=task_description, trial=trial,
                mode=mode, success=success, steps=step+1, replans=replans)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--adapter', type=Path, required=True)
    parser.add_argument('--task-report', type=Path, required=True)
    parser.add_argument('--source-report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--suites', default='libero_10,libero_goal,libero_spatial,libero_object')
    parser.add_argument('--max-tasks', type=int, default=3)
    parser.add_argument('--trials', type=int, default=5)
    parser.add_argument('--history-tokens', type=int, default=64)
    parser.add_argument('--resize-size', type=int, default=224)
    parser.add_argument('--replan-steps', type=int, default=5)
    parser.add_argument('--num-steps', type=int, default=10)
    parser.add_argument('--wait-steps', type=int, default=10)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--list-tasks', action='store_true')
    args = parser.parse_args()
    if min(args.max_tasks, args.trials, args.history_tokens, args.replan_steps,
           args.num_steps, args.wait_steps) < 1:
        parser.error('Counts must be positive')
    if args.replan_steps > 10:
        parser.error('replan-steps cannot exceed the action horizon')
    if args.output.exists() or args.output.with_suffix('.jsonl').exists():
        raise FileExistsError(args.output)
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    final_report = json.loads(args.task_report.read_text())
    source_report = json.loads(args.source_report.read_text())
    if final_report.get('status') != 'complete' or source_report.get('status') != 'complete':
        raise ValueError('Task and source reports must be complete')
    allowed = {case['task'] for case in final_report['cases']}
    suites = args.suites.split(',')
    if any(name not in SUITE_LIMITS for name in suites):
        raise ValueError(f'Unsupported suites: {suites}')
    tasks = select_tasks(benchmark, allowed, suites, args.max_tasks)
    selected = [dict(suite=name, task_id=i, language=task.language)
                for name, _, i, task in tasks]
    print('[rollout] selected tasks:', selected, flush=True)
    if args.list_tasks:
        return
    if not torch.cuda.is_available():
        raise RuntimeError('Rollout requires a GPU node')
    device = torch.device('cuda:0')
    model, policy = make_policy(args.checkpoint, device)
    loaded = torch.load(args.adapter, map_location='cpu', weights_only=True)
    layer = int(loaded['layer'])
    adapter = LayerMemory(loaded['embed_dim'], loaded['memory_dim']).to(device).eval()
    adapter.load_state_dict(loaded['adapter'], strict=True)
    norm_stats = normalize.load(args.checkpoint / 'assets/physical-intelligence/libero')
    cfg = config.get_config('pi05_libero').model
    output_transform = transforms.compose([
        *config.ModelTransformFactory()(cfg).outputs,
        # The shared stats also contain coarse_actions, which sampling does not emit.
        transforms.Unnormalize({'actions': norm_stats['actions']}, use_quantiles=True),
        libero_policy.LiberoOutputs(),
    ])
    source_case = source_report['cases'][0]
    fixed_episode = source_case['same_episode']
    fixed_frame = 10
    fixed_raw = raw_observation(rows_for(args.data_dir, fixed_episode)[fixed_frame], source_case['task'])
    fixed_obs, _ = observation(policy._input_transform, fixed_raw, device)
    fixed_features, fixed_mask = capture_history(model, fixed_obs, layer, args.history_tokens)
    fixed_state = adapter.build_state(fixed_features, fixed_mask)
    records = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.with_suffix('.jsonl').open('x') as stream:
        for suite_name, suite, task_id, task in tasks:
            # LIBERO's pinned initial-state files contain NumPy arrays; its
            # helper omits weights_only and fails under our PyTorch >=2.6.
            init_path = Path(get_libero_path('init_states')) / task.problem_folder / task.init_states_file
            init_states = torch.load(init_path, weights_only=False)
            if len(init_states) < args.trials:
                raise ValueError(f'Only {len(init_states)} initial states for {suite_name}:{task_id}')
            bddl = Path(get_libero_path('bddl_files')) / task.problem_folder / task.bddl_file
            env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256,
                                     camera_widths=256)
            try:
                for trial in range(args.trials):
                    for mode in MODES:
                        row = run_trial(args, env, init_states[trial], task.language, suite_name,
                                        task_id, trial, mode, model, policy, adapter, layer,
                                        fixed_state, output_transform, device)
                        records.append(row)
                        stream.write(json.dumps(row)+'\n')
                        stream.flush()
                        print(f"[rollout] {suite_name}:{task_id} trial={trial} mode={mode} "
                              f"success={row['success']} steps={row['steps']}", flush=True)
            finally:
                env.close()
    result = dict(status='complete', adapter=str(args.adapter), layer=layer,
                  selected_tasks=selected, fixed_history_source=dict(episode=fixed_episode,
                                                                       task=source_case['task'],
                                                                       frame=fixed_frame),
                  settings={k:str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
                  summary=summarize(records))
    args.output.write_text(json.dumps(result, indent=2))
    print(f'[done] {args.output}', flush=True)


if __name__ == '__main__':
    main()
