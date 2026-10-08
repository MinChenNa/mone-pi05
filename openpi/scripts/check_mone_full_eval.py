"""Merge four all-layer shards and compute paired task-level effects."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

if __package__:
    from .eval_mone_all_layers import MODES
    from .plan_mone_full import digest
else:
    from eval_mone_all_layers import MODES
    from plan_mone_full import digest


def effect(values, *, seed=1802, draws=10000):
    array = np.asarray(values, dtype=np.float64)
    if len(array) < 4 or not np.isfinite(array).all():
        raise ValueError('Need at least four finite paired task effects')
    rng = np.random.default_rng(seed)
    indices = rng.integers(len(array), size=(draws, len(array)))
    return dict(mean=float(array.mean()), ci95=np.quantile(array[indices].mean(1), [.025, .975]).tolist(),
                task_wins=int((array > 0).sum()), task_count=len(array))


def summarize(plan, reports, rows, split):
    expected_tasks = set(plan['partitions'][split])
    if set(task for report in reports for task in report['tasks']) != expected_tasks:
        raise ValueError('Evaluation task set differs from frozen plan')
    frames, draws = reports[0]['frames'], reports[0]['draws']
    seen = set()
    grouped = defaultdict(list)
    for row in rows:
        key = (row['task'], row['episode'], row['frame'], row['draw'])
        if key in seen or set(row['losses']) != set(MODES):
            raise ValueError(f'Duplicate or incomplete paired row: {key}')
        seen.add(key)
        if row['task'] not in expected_tasks:
            raise ValueError(f'Unexpected task: {row["task"]}')
        if not np.isfinite(list(row['losses'].values())).all():
            raise ValueError(f'Nonfinite loss: {key}')
        grouped[row['task']].append(row['losses'])
    if any(len(grouped[task]) != 2*frames*draws for task in expected_tasks):
        raise ValueError('Incomplete task evaluation')
    task_means = {task: {mode: float(np.mean([row[mode] for row in grouped[task]]))
                         for mode in MODES} for task in sorted(expected_tasks)}
    effects = {control: effect([task_means[task][control]-task_means[task]['relevant']
                                for task in sorted(expected_tasks)])
               for control in MODES if control != 'relevant'}
    primary = ('same_task', 'fixed_history', 'current_frame', 'parameter_control')
    verdict = ('offline_selective_signal' if
               all(effects[name]['ci95'][0] > 0 for name in (*primary, 'baseline')) and
               min(effects[name]['mean'] for name in primary) >= .001
               else 'inconclusive')
    return dict(status='complete', split=split, tasks=len(expected_tasks), samples=len(rows),
                verdict=verdict, task_means=task_means, effects=effects,
                thresholds=dict(min_mean_gain=.001, task_bootstrap_lower_bound=0),
                note='Paired offline 7D flow loss. Closed-loop success remains the final gate.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--split', choices=('val', 'test'), required=True)
    parser.add_argument('--inputs', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or len(args.inputs) != 4:
        raise ValueError('Output must be new and exactly four shards are required')
    plan = json.loads(args.plan.read_text())
    plan_hash = plan.pop('plan_sha256')
    if digest(plan) != plan_hash:
        raise ValueError('Plan hash mismatch')
    reports = [json.loads(path.read_text()) for path in args.inputs]
    if {report['shard'] for report in reports} != set(range(4)):
        raise ValueError('Missing or duplicate shards')
    common = {(report['status'], report['split'], report['plan_sha256'], report['shards'],
               report['frames'], report['draws'], report['backbone_sha256'],
               report['memory_adapter'], report['control_adapter'], report['checkpoint'])
              for report in reports}
    if len(common) != 1 or next(iter(common))[:4] != ('complete', args.split, plan_hash, 4):
        raise ValueError('Shard provenance/settings mismatch')
    rows = []
    for path, report in zip(args.inputs, reports, strict=True):
        shard_rows = [json.loads(line) for line in path.with_suffix('.jsonl').read_text().splitlines()]
        if len(shard_rows) != report['records']:
            raise ValueError(f'JSONL count mismatch: {path}')
        rows.extend(shard_rows)
    result = summarize(plan, reports, rows, args.split)
    result['sources'] = [str(path) for path in args.inputs]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(f"[eval] split={args.split} verdict={result['verdict']} report={args.output}")
    for name, item in result['effects'].items():
        print(f"[paired] control={name} gain={item['mean']:.6f} "
              f"ci95={item['ci95']} wins={item['task_wins']}/{item['task_count']}")


if __name__ == '__main__':
    main()
