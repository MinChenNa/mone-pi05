"""Validate all shards and summarize the preregistered history-content screen."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

if __package__:
    from .eval_mone_history_screen import CONDITIONS, MODES, canonical_hash
else:
    from eval_mone_history_screen import CONDITIONS, MODES, canonical_hash


def effect(values, seed=4107, draws=10000):
    array = np.asarray(values, dtype=np.float64)
    if len(array) == 0 or not np.isfinite(array).all():
        raise ValueError('Need finite task-level paired effects')
    rng = np.random.default_rng(seed)
    indices = rng.integers(len(array), size=(draws, len(array)))
    return dict(mean=float(array.mean()), ci95=np.quantile(array[indices].mean(1), [.025, .975]).tolist(),
                task_wins=int((array > 0).sum()), task_count=len(array))


def summarize(plan, records, split):
    cases = [case for case in plan['cases'] if case['split'] == split]
    expected = {(case['task'], case['episode'], frame, condition, draw)
                for case in cases for frame in case['frames']
                for condition in CONDITIONS for draw in range(plan['settings']['draws'])}
    received = {}
    for row in records:
        key = (row['task'], row['episode'], row['frame'], row['condition'], row['draw'])
        if key in received or set(row['losses']) != set(MODES):
            raise ValueError(f'Duplicate or incomplete row: {key}')
        if not all(np.isfinite(v) for v in row['losses'].values()):
            raise ValueError(f'Nonfinite loss: {key}')
        received[key] = row
    if received.keys() != expected:
        raise ValueError(f'Missing {len(expected - received.keys())}, extra {len(received.keys() - expected)} rows')
    grouped = defaultdict(list)
    for row in records:
        grouped[(row['task'], row['condition'])].append(row['losses'])
    tasks = sorted({case['task'] for case in cases})
    means = {condition: {task: {mode: float(np.mean([r[mode] for r in grouped[task, condition]]))
                                for mode in MODES} for task in tasks} for condition in CONDITIONS}
    effects = {}
    for condition in CONDITIONS:
        effects[condition] = {control: effect([means[condition][task][control] -
                                              means[condition][task]['relevant'] for task in tasks])
                              for control in MODES if control != 'relevant'}
    sensitivity = {
        condition: effect([means[condition][task]['baseline'] - means['clean'][task]['baseline']
                           for task in tasks]) for condition in CONDITIONS if condition != 'clean'
    }
    primary = effects['center_occluded']
    primary_controls = ('baseline', 'same_task', 'fixed_history', 'current_frame')
    selective = all(primary[name]['ci95'][0] > 0 for name in primary_controls)
    practical = primary['same_task']['mean'] >= 0.001 and primary['same_task']['task_wins'] >= 9
    informative_mask = sensitivity['center_occluded']['mean'] > 0.001
    baseline_clean = np.mean([means['clean'][task]['baseline'] for task in tasks])
    clean_degradation = effects['clean']['baseline']['mean'] < -0.01 * baseline_clean
    verdict = ('screen_positive' if selective and practical and informative_mask and not clean_degradation
               else 'inconclusive')
    return dict(status='complete', split=split, plan_sha256=plan['plan_sha256'],
                tasks=len(tasks), episodes=len(cases), samples=len(expected),
                verdict=verdict, task_means=means, effects=effects,
                baseline_mask_sensitivity=sensitivity,
                criteria=dict(primary='center_occluded', min_same_task_gain=0.001,
                              min_task_wins=9, lower_ci_positive_for=list(primary_controls),
                              min_baseline_mask_sensitivity=0.001,
                              max_clean_degradation_fraction=0.01),
                note='One adapter training seed. Exploratory offline action7 loss, not rollout success.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--split', choices=('screen', 'confirm'), required=True)
    parser.add_argument('--inputs', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    plan = json.loads(args.plan.read_text())
    if canonical_hash({k: v for k, v in plan.items() if k != 'plan_sha256'}) != plan['plan_sha256']:
        raise ValueError('Plan hash mismatch')
    if len(args.inputs) != 4:
        raise ValueError('Expected four GPU shard reports')
    records = []
    shards = set()
    for path in args.inputs:
        report = json.loads(path.read_text())
        if (report['status'], report['split'], report['plan_sha256'], report['shards'], report['smoke']) != (
            'complete', args.split, plan['plan_sha256'], 4, False):
            raise ValueError(f'Shard metadata mismatch: {path}')
        if report['shard'] in shards:
            raise ValueError(f'Duplicate shard: {report["shard"]}')
        shards.add(report['shard'])
        rows = [json.loads(line) for line in path.with_suffix('.jsonl').read_text().splitlines()]
        if len(rows) != report['records']:
            raise ValueError(f'JSONL count mismatch: {path}')
        records.extend(rows)
    if shards != set(range(4)):
        raise ValueError(f'Missing shards: {set(range(4)) - shards}')
    result = summarize(plan, records, args.split)
    result['sources'] = [str(path) for path in args.inputs]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(f"[history-screen] split={args.split} verdict={result['verdict']} "
          f"tasks={result['tasks']} samples={result['samples']} report={args.output}")
    for name, item in result['effects']['center_occluded'].items():
        print(f"[center] control={name} gain={item['mean']:.6f} "
              f"ci95={item['ci95']} wins={item['task_wins']}/{item['task_count']}")


if __name__ == '__main__':
    main()
