"""Frozen-adapter screen for whether the correct t-10 frame matters."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

if __package__:
    from .eval_mone_heldout import observation, rows_for, TRAIN_EPISODES
    from .train_mone_adapter_contrastive import make_policy, raw_observation
else:
    from eval_mone_heldout import observation, rows_for, TRAIN_EPISODES
    from train_mone_adapter_contrastive import make_policy, raw_observation
from openpi.models_pytorch.mone_layer import LayerMemory, capture_history


CONDITIONS = ('clean', 'no_wrist', 'center_occluded')
MODES = ('baseline', 'relevant', 'same_task', 'fixed_history', 'current_frame')
TAG = '20260926-124724-16706'
DEFAULT_REPORT = Path(f'../outputs/heldout/L4-selective-unused12-{TAG}.json')
DEFAULT_TRAIN = Path(f'../outputs/mone_selective_L4_{TAG}.json')
DEFAULT_ADAPTER = Path(f'../outputs/mone_selective_L4_{TAG}.pt')


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def metadata_for(data_dir):
    return [json.loads(line) for line in (data_dir / 'meta/episodes.jsonl').read_text().splitlines()]


def episode_path(data_dir, episode):
    info = json.loads((data_dir / 'meta/info.json').read_text())
    pattern = info.get('data_path', 'data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet')
    return data_dir / pattern.format(episode_chunk=episode // info.get('chunks_size', 1000),
                                     episode_index=episode)


def make_plan(data_dir, report, train, adapter, seed=92026):
    if report.get('status') != 'complete' or train.get('status') != 'complete':
        raise ValueError('Source evaluation and training reports must be complete')
    tasks = sorted({case['task'] for case in report['cases']})
    if len(tasks) != 12 or set(tasks) & set(train['tasks']):
        raise ValueError('Expected 12 tasks not used for adapter training')
    if hashlib.sha256(adapter.read_bytes()).hexdigest() != report['adapter_sha256']:
        raise ValueError('Adapter hash differs from source evaluation')
    if train['layer'] != 4:
        raise ValueError('Expected L4 adapter')
    previous = set(TRAIN_EPISODES) | set(train['episodes'])
    for case in report['cases']:
        previous.update(case[key] for key in ('episode', 'same_episode', 'wrong_episode'))
    by_task = defaultdict(list)
    metadata = metadata_for(data_dir)
    lengths = {item['episode_index']: item['length'] for item in metadata}
    for item in metadata:
        if item['length'] >= 30 and item['episode_index'] not in previous:
            by_task[item['tasks'][0]].append(item['episode_index'])
    rng = np.random.default_rng(seed)
    picks = {}
    for task in tasks:
        if len(by_task[task]) < 5:
            raise ValueError(f'Need 5 fresh episodes for {task}; found {len(by_task[task])}')
        picks[task] = rng.permutation(sorted(by_task[task]))[:5].tolist()
    # The fixed donor is outside all selected tasks and all old evaluation episodes.
    fixed_candidates = sorted((task, episode) for task, episodes in by_task.items()
                              if task not in tasks and task not in train['tasks']
                              for episode in episodes)
    if not fixed_candidates:
        raise ValueError('No disjoint fixed-history donor available')
    fixed_task, fixed_episode = fixed_candidates[seed % len(fixed_candidates)]
    cases = []
    for index, task in enumerate(tasks):
        wrong_task = tasks[(index + 1) % len(tasks)]
        for split, episodes in (('screen', picks[task][:2]), ('confirm', picks[task][2:4])):
            for episode in episodes:
                frames = sorted(set(np.linspace(10, lengths[episode]-10, 5).round().astype(int).tolist()))
                if len(frames) != 5:
                    raise ValueError(f'Could not select five unique frames for episode {episode}')
                cases.append(dict(split=split, task=task, episode=episode, frames=frames,
                                  same_episode=picks[task][4], wrong_episode=picks[wrong_task][4]))
    selected = {case[key] for case in cases
                for key in ('episode', 'same_episode', 'wrong_episode')} | {fixed_episode}
    for episode in selected:
        if not episode_path(data_dir, episode).is_file():
            raise FileNotFoundError(episode_path(data_dir, episode))
    plan = dict(status='prepared', seed=seed, task_count=len(tasks), cases=cases,
                fixed_history_source=dict(task=fixed_task, episode=fixed_episode, frame=10),
                source_adapter_sha256=report['adapter_sha256'], layer=4,
                settings=dict(frames_per_episode=5, draws=3, history_lag=10,
                              history_tokens=64, center_area_fraction=0.5,
                              modes=MODES, conditions=CONDITIONS),
                interpretation='Exploratory: tasks appeared in a previous evaluation; episodes and donors are fresh.')
    plan['plan_sha256'] = canonical_hash(plan)
    return plan


def mask_current(raw, condition):
    if condition not in CONDITIONS:
        raise ValueError(condition)
    result = dict(raw)
    if condition == 'no_wrist':
        result['observation/wrist_image'] = np.full_like(raw['observation/wrist_image'], 127)
    elif condition == 'center_occluded':
        for key in ('observation/image', 'observation/wrist_image'):
            image = raw[key].copy()
            height, width = image.shape[:2]
            side = int(round(min(height, width) * np.sqrt(0.5)))
            top, left = (height - side) // 2, (width - side) // 2
            image[top:top+side, left:left+side] = 127
            result[key] = image
    return result


def donor_frame(frame, current_length, donor_length):
    return min(donor_length - 1, round((frame - 10) / (current_length - 1) * (donor_length - 1)))


@torch.no_grad()
def evaluate(args, plan, cases):
    if not torch.cuda.is_available():
        raise RuntimeError('A CUDA GPU is required to evaluate the backbone')
    device = torch.device('cuda:0')
    model, policy = make_policy(args.checkpoint, device)
    loaded = torch.load(args.adapter, map_location='cpu', weights_only=True)
    if loaded['layer'] != plan['layer']:
        raise ValueError('Adapter layer differs from plan')
    adapter = LayerMemory(loaded['embed_dim'], loaded['memory_dim']).to(device).eval()
    adapter.load_state_dict(loaded['adapter'], strict=True)

    def state_for(row, prompt):
        past, _ = observation(policy._input_transform, raw_observation(row, prompt), device)
        features, mask = capture_history(model, past, plan['layer'], args.history_tokens)
        return adapter.build_state(features, mask)

    fixed = plan['fixed_history_source']
    fixed_rows = rows_for(args.data_dir, fixed['episode'])
    fixed_state = state_for(fixed_rows[fixed['frame']], fixed['task'])
    records = []
    with args.output.with_suffix('.jsonl').open('x') as stream:
        for case in cases:
            rows = rows_for(args.data_dir, case['episode'])
            same_rows = rows_for(args.data_dir, case['same_episode'])
            if model.config.action_horizon != 10:
                raise ValueError('Plan assumes action horizon 10')
            for frame in case['frames'][:1] if args.smoke else case['frames']:
                target = np.asarray([r['actions'] for r in rows[frame:frame+model.config.action_horizon]],
                                    dtype=np.float32)
                same_frame = donor_frame(frame, len(rows), len(same_rows))
                history_states = {
                    'relevant': state_for(rows[frame-10], case['task']),
                    'same_task': state_for(same_rows[same_frame], case['task']),
                    'fixed_history': fixed_state,
                }
                unmasked = raw_observation(rows[frame], case['task'])
                for condition in CONDITIONS:
                    raw = mask_current(unmasked, condition)
                    raw['actions'] = target
                    current, actions = observation(policy._input_transform, raw, device)
                    current_features, current_mask = capture_history(
                        model, current, plan['layer'], args.history_tokens)
                    history_states['current_frame'] = adapter.build_state(current_features, current_mask)
                    for draw in range(args.draws):
                        seed = args.seed + case['episode'] * 10000 + int(frame) * 10 + draw
                        generator = torch.Generator(device=device).manual_seed(seed)
                        noise = torch.randn(actions.shape, device=device, dtype=actions.dtype,
                                            generator=generator)
                        time = torch.tensor([(draw + 0.5) / args.draws], device=device, dtype=torch.float32)
                        losses = {}
                        for mode in MODES:
                            if mode == 'baseline':
                                value = model(current, actions, noise=noise, time=time,
                                              preprocess_train=False)[..., :7].mean()
                            else:
                                with adapter.activate(model, history_states[mode], plan['layer']):
                                    value = model(current, actions, noise=noise, time=time,
                                                  preprocess_train=False)[..., :7].mean()
                            if not torch.isfinite(value):
                                raise RuntimeError(f'Nonfinite loss: {case}, frame={frame}, {condition}, {mode}')
                            losses[mode] = float(value)
                        record = dict(task=case['task'], episode=case['episode'], frame=int(frame),
                                      split=case['split'], condition=condition, draw=draw, seed=seed,
                                      history_episode=case['episode'], history_frame=int(frame-10),
                                      same_episode=case['same_episode'], same_frame=same_frame,
                                      losses=losses)
                        records.append(record)
                        stream.write(json.dumps(record) + '\n')
                        stream.flush()
                print(f"[screen] task={case['task']} episode={case['episode']} frame={frame}", flush=True)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'evaluate'))
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--adapter', type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument('--source-report', type=Path, default=DEFAULT_REPORT)
    parser.add_argument('--train-report', type=Path, default=DEFAULT_TRAIN)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--split', choices=('screen', 'confirm'))
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=4)
    parser.add_argument('--frames', type=int, default=5)
    parser.add_argument('--draws', type=int, default=3)
    parser.add_argument('--history-tokens', type=int, default=64)
    parser.add_argument('--seed', type=int, default=20260926)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.command == 'prepare':
        if args.plan.exists():
            raise FileExistsError(args.plan)
        result = make_plan(args.data_dir, json.loads(args.source_report.read_text()),
                           json.loads(args.train_report.read_text()), args.adapter)
        args.plan.parent.mkdir(parents=True, exist_ok=True)
        args.plan.write_text(json.dumps(result, indent=2))
        print(f"[prepared] {args.plan} tasks={result['task_count']} plan_sha256={result['plan_sha256']}")
        return
    if args.output is None or args.split is None:
        parser.error('evaluate requires --output and --split')
    if min(args.frames, args.draws, args.history_tokens, args.shards) < 1 or not 0 <= args.shard < args.shards:
        parser.error('Invalid frame/draw/token/shard count')
    if args.output.exists() or args.output.with_suffix('.jsonl').exists():
        raise FileExistsError(args.output)
    plan = json.loads(args.plan.read_text())
    plan_hash = plan.pop('plan_sha256')
    if canonical_hash(plan) != plan_hash:
        raise ValueError('Plan changed after preparation')
    if hashlib.sha256(args.adapter.read_bytes()).hexdigest() != plan['source_adapter_sha256']:
        raise ValueError('Adapter changed after preparation')
    if (args.frames, args.draws, args.history_tokens) != (
        plan['settings']['frames_per_episode'], plan['settings']['draws'], plan['settings']['history_tokens']
    ) and not args.smoke:
        raise ValueError('Evaluation settings differ from preregistered plan')
    tasks = sorted({case['task'] for case in plan['cases']})
    allowed = set(tasks[args.shard::args.shards])
    cases = [case for case in plan['cases'] if case['split'] == args.split and case['task'] in allowed]
    if args.smoke:
        cases = cases[:1]
    if not cases:
        raise ValueError('No cases assigned to this shard')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    records = evaluate(args, plan, cases)
    result = dict(status='complete', plan_sha256=plan_hash, split=args.split,
                  shard=args.shard, shards=args.shards, smoke=args.smoke,
                  frames=args.frames, draws=args.draws, cases=cases, records=len(records),
                  task_count=len(allowed), adapter_sha256=plan['source_adapter_sha256'])
    args.output.write_text(json.dumps(result, indent=2))
    print(f'[done] {args.output} records={len(records)}')


if __name__ == '__main__':
    main()
