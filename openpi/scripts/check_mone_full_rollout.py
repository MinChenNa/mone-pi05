"""Merge suite rollouts and compare paired success at task and trial level."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

if __package__:
    from .eval_mone_full_rollout import MODES
    from .plan_mone_full import digest
else:
    from eval_mone_full_rollout import MODES
    from plan_mone_full import digest


def task_bootstrap(values, seed=1904, draws=10000):
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 4:
        raise ValueError('Need at least four tasks')
    rng = np.random.default_rng(seed)
    indices = rng.integers(len(values), size=(draws, len(values)))
    return dict(mean=float(values.mean()), ci95=np.quantile(values[indices].mean(1), [.025,.975]).tolist(),
                task_wins=int((values > 0).sum()), tasks=len(values))


def summarize(plan, reports, records):
    expected_tasks = set(plan['partitions']['test'])
    if set(task for report in reports for task in report['tasks']) != expected_tasks:
        raise ValueError('Rollout test tasks differ from frozen plan')
    paired = defaultdict(dict)
    for row in records:
        key = (row['task'], row['trial'])
        if row['mode'] in paired[key]:
            raise ValueError(f'Duplicate rollout mode for {key}')
        paired[key][row['mode']] = int(row['success'])
    trials = reports[0]['trials']
    if len(paired) != len(expected_tasks)*trials or any(set(item) != set(MODES) for item in paired.values()):
        raise ValueError('Incomplete paired rollout trials')
    by_task = {task: {mode: float(np.mean([paired[task, trial][mode] for trial in range(trials)]))
                      for mode in MODES} for task in sorted(expected_tasks)}
    rates = {mode: float(np.mean([row[mode] for row in paired.values()])) for mode in MODES}
    comparisons = {}
    for control in MODES:
        if control == 'relevant':
            continue
        deltas = [row['relevant']-row[control] for row in paired.values()]
        comparisons[control] = dict(
            task_effect=task_bootstrap([by_task[task]['relevant']-by_task[task][control]
                                        for task in sorted(expected_tasks)]),
            relevant_only=sum(value > 0 for value in deltas),
            control_only=sum(value < 0 for value in deltas),
            ties=sum(value == 0 for value in deltas))
    primary = ('baseline', 'same_task', 'fixed_history', 'parameter_control')
    verdict = ('success_signal' if rates['baseline'] > 0 and
               all(comparisons[name]['task_effect']['ci95'][0] > 0 for name in primary) and
               min(comparisons[name]['task_effect']['mean'] for name in primary) >= .05
               else 'inconclusive')
    return dict(status='complete', verdict=verdict, paired_trials=len(paired),
                task_count=len(expected_tasks), success_rate=rates,
                by_task=by_task, paired=comparisons,
                note='Task-bootstrap descriptive CI, one trained seed. A ceiling baseline may limit positive gains.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or len(args.inputs) != 4:
        raise ValueError('Output must be new and four suite reports are required')
    plan = json.loads(args.plan.read_text())
    plan_hash = plan.pop('plan_sha256')
    if digest(plan) != plan_hash:
        raise ValueError('Plan hash mismatch')
    reports = [json.loads(path.read_text()) for path in args.inputs]
    if {report['suite'] for report in reports} != {
        'libero_spatial', 'libero_object', 'libero_goal', 'libero_10'}:
        raise ValueError('Missing or duplicate LIBERO suite')
    common = {(r['status'], r['plan_sha256'], r['trials'], r['backbone_sha256'],
               r['memory_adapter'], r['control_adapter'], r['checkpoint']) for r in reports}
    if len(common) != 1 or next(iter(common))[:2] != ('complete', plan_hash):
        raise ValueError('Rollout provenance/settings mismatch')
    records = []
    for path, report in zip(args.inputs, reports, strict=True):
        rows = [json.loads(line) for line in path.with_suffix('.jsonl').read_text().splitlines()]
        if len(rows) != report['records']:
            raise ValueError(f'JSONL count mismatch: {path}')
        records.extend(rows)
    result = summarize(plan, reports, records)
    result['sources'] = [str(path) for path in args.inputs]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(f"[rollout] verdict={result['verdict']} trials={result['paired_trials']} "
          f"success_rate={result['success_rate']} report={args.output}")


if __name__ == '__main__':
    main()
