"""Predeclared offline selectivity gate for paired old/new L4 evaluations."""

import argparse
import json
from pathlib import Path

import numpy as np


CONTROLS = ('baseline', 'wrong_task', 'same_task', 'current_frame', 'fixed_history')


def paired_effect(values, seed=719, draws=10000):
    values = np.asarray(values, dtype=np.float64)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError('Paired task effects must be finite and nonempty')
    rng = np.random.default_rng(seed)
    indices = rng.integers(len(values), size=(draws, len(values)))
    return dict(mean=float(values.mean()), ci95=np.quantile(values[indices].mean(1), [.025,.975]).tolist(),
                task_win_fraction=float((values > 0).mean()))


def assess(old, new, train, min_gain):
    if old.get('status') != 'complete' or new.get('status') != 'complete' or train.get('status') != 'complete':
        raise ValueError('Both evaluations and training must be complete')
    if old['cases'] != new['cases']:
        raise ValueError('Old/new evaluations used different cases')
    old_tasks = old['summary_action7_task']['per_task']
    new_tasks = new['summary_action7_task']['per_task']
    if old_tasks.keys() != new_tasks.keys():
        raise ValueError('Old/new evaluations used different task sets')
    if set(old_tasks) & set(train['tasks']):
        raise ValueError('Final evaluation contains adapter-training tasks')
    for task in old_tasks:
        if abs(old_tasks[task]['baseline'] - new_tasks[task]['baseline']) > 1e-5:
            raise ValueError(f'Backbone baseline changed for task {task}')
    tasks = sorted(new_tasks)
    effects = {name: paired_effect([new_tasks[task][name]-new_tasks[task]['relevant']
                                    for task in tasks]) for name in CONTROLS}
    effects['old_relevant'] = paired_effect([old_tasks[task]['relevant']-new_tasks[task]['relevant']
                                            for task in tasks])
    control_names = tuple(name for name in CONTROLS if name != 'baseline')
    positive = all(effects[name]['ci95'][0] > 0 for name in CONTROLS)
    practical = min(effects[name]['mean'] for name in control_names) >= min_gain
    retains_quality = effects['old_relevant']['mean'] >= 0
    if positive and practical and retains_quality:
        verdict = 'pass_offline_selectivity'
    elif any(effects[name]['mean'] <= 0 for name in control_names) or not retains_quality:
        verdict = 'fail_current_adapter'
    else:
        verdict = 'inconclusive'
    return dict(verdict=verdict, task_count=len(tasks), min_mean_control_gain=min_gain,
                effects=effects,
                interpretation='Paired offline action7 flow loss only; pass is not rollout success.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old', type=Path, required=True)
    parser.add_argument('--new', type=Path, required=True)
    parser.add_argument('--train', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--min-mean-control-gain', type=float, default=0.001)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = assess(json.loads(args.old.read_text()), json.loads(args.new.read_text()),
                    json.loads(args.train.read_text()), args.min_mean_control_gain)
    result['sources'] = dict(old=str(args.old), new=str(args.new), train=str(args.train))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(f"[gate1] {result['verdict']} report={args.output}", flush=True)
    for name, item in result['effects'].items():
        print(f"[gate1] {name} mean={item['mean']:.6f} ci95={item['ci95']}", flush=True)


if __name__ == '__main__':
    main()
