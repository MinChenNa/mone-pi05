"""Replay only the three observed regressions with trajectories and action guards.

This is a post-hoc diagnostic, not a new held-out benchmark result. The
original full18 reports and checkpoints remain untouched.
"""

import argparse
from collections import deque
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

if __package__:
    from .eval_mone_all_layers import file_sha256, load_adapter
    from .eval_mone_heldout import observation
    from .eval_mone_rollout import SUITE_LIMITS, WAIT_ACTION, sim_observation
    from .plan_mone_full import digest
    from .train_mone_adapter_contrastive import make_policy
else:
    from eval_mone_all_layers import file_sha256, load_adapter
    from eval_mone_heldout import observation
    from eval_mone_rollout import SUITE_LIMITS, WAIT_ACTION, sim_observation
    from plan_mone_full import digest
    from train_mone_adapter_contrastive import make_policy
from openpi import transforms
from openpi.models_pytorch.mone_all_layers import capture_context_layers
from openpi.models_pytorch.mone_guard import cap_action_delta
from openpi.policies import libero_policy
from openpi.shared import normalize
from openpi.training import config


def regression_cases(paired_report):
    records = []
    for source in paired_report['sources']:
        records.extend(json.loads(line) for line in Path(source).with_suffix('.jsonl').read_text().splitlines())
    grouped = {}
    for row in records:
        key = (row['suite'], row['task_id'], row['task'], row['trial'])
        grouped.setdefault(key, {})[row['mode']] = row
    cases = []
    for (suite, task_id, task, trial), modes in grouped.items():
        if modes['baseline']['success'] and not modes['relevant']['success']:
            cases.append(dict(suite=suite, task_id=task_id, task=task, trial=trial,
                              original={name: dict(success=bool(row['success']), steps=row['steps'])
                                        for name, row in modes.items()}))
    return sorted(cases, key=lambda item: (item['suite'], item['task_id'], item['trial']))


def small_state(obs):
    result = {}
    for key, value in obs.items():
        array = np.asarray(value)
        if array.size <= 16 and array.dtype.kind in 'fiub':
            result[key] = array.astype(np.float64).tolist()
    return result


def mode_name(radius):
    return 'guarded_' + format(radius, '.3f').replace('.', 'p')


@torch.no_grad()
def replay(args, env, initial_state, case, mode, radius, model, policy, adapter,
           output_transform, device, trace_path, image_dir):
    env.seed(args.seed + case['trial'])
    env.reset()
    obs = env.set_init_state(initial_state)
    history = deque(maxlen=11)
    history.append(sim_observation(obs, case['task'], args.resize_size))
    for _ in range(args.wait_steps):
        obs, _, _, _ = env.step(WAIT_ACTION)
        history.append(sim_observation(obs, case['task'], args.resize_size))
    plan = deque()
    replans = 0
    max_deltas = []
    guard_scales = []
    success = False
    image_dir.mkdir(parents=True, exist_ok=False)
    with trace_path.open('x') as stream:
        for step in range(SUITE_LIMITS[case['suite']]):
            if not plan:
                current, _ = observation(policy._input_transform, history[-1], device)
                generator = torch.Generator(device=device).manual_seed(
                    args.seed + case['task_id']*100000 + case['trial']*1000 + replans)
                noise = torch.randn((1, model.config.action_horizon, model.config.action_dim),
                                    generator=generator, device=device, dtype=torch.float32)
                baseline = model.sample_actions(device, current, noise=noise,
                                                num_steps=args.num_steps)
                proposed = None
                if mode != 'baseline':
                    past, _ = observation(policy._input_transform, history[0], device)
                    features, mask = capture_context_layers(model, past, args.history_tokens)
                    states = adapter.build_states(features, mask)
                    with adapter.activate(model, states):
                        proposed = model.sample_actions(device, current, noise=noise,
                                                        num_steps=args.num_steps)
                if mode == 'baseline':
                    actions = baseline
                    scale = maximum = None
                elif mode == 'relevant':
                    actions = proposed
                    scale = None
                    maximum = float((proposed[..., :7]-baseline[..., :7]).abs().max())
                else:
                    actions, scales, maxima = cap_action_delta(baseline, proposed, radius)
                    scale = float(scales[0])
                    maximum = float(maxima[0])
                    guard_scales.append(scale)
                if maximum is not None:
                    max_deltas.append(maximum)
                outputs = output_transform(dict(state=current.state[0].detach().cpu().numpy(),
                                                actions=actions[0].detach().float().cpu().numpy()))
                chunk = np.asarray(outputs['actions'], dtype=np.float64)
                if chunk.shape != (model.config.action_horizon, 7) or not np.isfinite(chunk).all():
                    raise ValueError(f'Invalid LIBERO action chunk: {chunk.shape}')
                plan.extend(chunk[:args.replan_steps])
                stream.write(json.dumps(dict(kind='replan', step=step, replan=replans,
                                             mode=mode, max_normalized_delta=maximum,
                                             guard_scale=scale,
                                             baseline_actions=baseline[0, :args.replan_steps, :7].float().cpu().tolist(),
                                             proposed_actions=(None if proposed is None else
                                                               proposed[0, :args.replan_steps, :7].float().cpu().tolist()),
                                             executed_actions=chunk[:args.replan_steps].tolist()))+'\n')
                replans += 1
            action = plan.popleft()
            obs, _, done, _ = env.step(action.tolist())
            history.append(sim_observation(obs, case['task'], args.resize_size))
            if step % args.frame_every == 0 or done:
                Image.fromarray(history[-1]['observation/image']).save(image_dir/f'step{step+1:04d}.png')
            stream.write(json.dumps(dict(kind='step', step=step+1, action=action.tolist(),
                                         state=small_state(obs), done=bool(done)))+'\n')
            if done:
                success = True
                break
    return dict(mode=mode, radius=radius, success=success, steps=step+1,
                replans=replans, mean_unguarded_max_delta=(float(np.mean(max_deltas))
                                                      if max_deltas else None),
                mean_guard_scale=(float(np.mean(guard_scales)) if guard_scales else None),
                trace=str(trace_path), images=str(image_dir))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--memory-adapter', type=Path, required=True)
    parser.add_argument('--paired-report', type=Path, required=True)
    parser.add_argument('--suite', choices=tuple(SUITE_LIMITS), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--radii', type=float, nargs='+', default=(0, .025, .05, .1, .2))
    parser.add_argument('--frame-every', type=int, default=20)
    parser.add_argument('--history-tokens', type=int, default=32)
    parser.add_argument('--wait-steps', type=int, default=10)
    parser.add_argument('--replan-steps', type=int, default=5)
    parser.add_argument('--num-steps', type=int, default=10)
    parser.add_argument('--resize-size', type=int, default=224)
    parser.add_argument('--seed', type=int, default=7)
    args = parser.parse_args()
    if (min(args.frame_every, args.history_tokens, args.wait_steps, args.replan_steps,
            args.num_steps) < 1 or args.replan_steps > 10 or
            any(not np.isfinite(value) or value < 0 for value in args.radii)):
        parser.error('Invalid replay settings')
    if len(set(args.radii)) != len(args.radii):
        parser.error('Duplicate trust radii')
    if args.output.exists() or args.output.with_suffix('').exists():
        raise FileExistsError(args.output)
    if not torch.cuda.is_available():
        raise RuntimeError('Replay requires a CUDA GPU')
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    paired = json.loads(args.paired_report.read_text())
    if paired.get('status') != 'complete' or paired.get('paired_trials') != 80:
        raise ValueError('Need the completed frozen 80-pair rollout report')
    cases = [case for case in regression_cases(paired) if case['suite'] == args.suite]
    if len(cases) != 1:
        raise ValueError(f'Expected exactly one regression in {args.suite}, found {len(cases)}')
    plan = json.loads(args.plan.read_text())
    plan_hash = plan.pop('plan_sha256')
    if digest(plan) != plan_hash:
        raise ValueError('Frozen data plan hash mismatch')
    device = torch.device('cuda:0')
    model, policy = make_policy(args.checkpoint, device)
    backbone_hash = file_sha256(args.checkpoint/'model.safetensors')
    adapter, saved = load_adapter(args.memory_adapter, plan_hash, backbone_hash, device)
    if saved['history_mode'] != 'true_history' or saved['history_tokens'] != args.history_tokens:
        raise ValueError('Wrong Mone adapter or history token count')
    cfg = config.get_config('pi05_libero').model
    norm_stats = normalize.load(args.checkpoint/'assets/physical-intelligence/libero')
    output_transform = transforms.compose([
        *config.ModelTransformFactory()(cfg).outputs,
        transforms.Unnormalize({'actions': norm_stats['actions']}, use_quantiles=True),
        libero_policy.LiberoOutputs(),
    ])
    suite = benchmark.get_benchmark_dict()[args.suite]()
    case = cases[0]
    task = suite.get_task(case['task_id'])
    if task.language != case['task'] or case['task'] not in plan['partitions']['test']:
        raise ValueError('Replay target differs from frozen test plan')
    init_path = Path(get_libero_path('init_states'))/task.problem_folder/task.init_states_file
    init_states = torch.load(init_path, weights_only=False)
    bddl = Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    trace_root = args.output.with_suffix('')
    trace_root.mkdir(exist_ok=False)
    results = []
    try:
        for mode, radius in [('baseline', None), ('relevant', None),
                             *((mode_name(value), value) for value in args.radii)]:
            trace = trace_root/f'{mode}.jsonl'
            images = trace_root/mode
            result = replay(args, env, init_states[case['trial']], case, mode, radius,
                            model, policy, adapter, output_transform, device, trace, images)
            results.append(result)
            print(f"[replay] {args.suite} trial={case['trial']} mode={mode} "
                  f"success={result['success']} steps={result['steps']} "
                  f"max_delta={result['mean_unguarded_max_delta']}", flush=True)
            if mode in ('baseline', 'relevant'):
                original = case['original'][mode]
                if (result['success'], result['steps']) != (original['success'], original['steps']):
                    raise RuntimeError(f'{mode} replay did not reproduce frozen result: '
                                       f'{result} vs {original}')
            if radius == 0:
                original = case['original']['baseline']
                if (result['success'], result['steps']) != (original['success'], original['steps']):
                    raise RuntimeError('Zero-radius guard did not reproduce frozen pi0.5 baseline')
    finally:
        env.close()
    args.output.write_text(json.dumps(dict(status='complete', diagnostic_only=True,
                                           case=case, results=results,
                                           backbone_sha256=backbone_hash,
                                           memory_adapter=str(args.memory_adapter),
                                           note='Post-hoc replay of observed failures; not independent evidence of improvement.'),
                                indent=2))
    print(f'[done] {args.output}', flush=True)


if __name__ == '__main__':
    main()
