"""Paired task-8 MoNe history-input ablation; no pi0.5 baseline is run."""

import argparse
import json
import math
from pathlib import Path

import numpy as np


MODES = ('relevant', 'static_initial', 'current_frame', 'same_task')


def load_run(path, mode):
    report = json.loads(path.read_text())
    if (report.get('status') != 'complete' or report.get('task_id') != 8 or
            report.get('trials') != 50 or report.get('seed') != 7):
        raise ValueError(f'Incomplete or incompatible task-8 run: {path}')
    if report.get('memory_input_mode', 'relevant') != mode:
        raise ValueError(f'Mode metadata mismatch: {path}')
    rows = [json.loads(line) for line in path.with_suffix('.jsonl').read_text().splitlines()]
    if len(rows) != 50 or [row['trial'] for row in rows] != list(range(50)):
        raise ValueError(f'Expected exactly trials 0..49: {path}')
    if any(row['mode'] != mode or row['task_id'] != 8 for row in rows):
        raise ValueError(f'Unexpected mode or task in {path}')
    if sum(bool(row['success']) for row in rows) != report['successes']:
        raise ValueError(f'Summary and episodes disagree: {path}')
    return report, np.array([bool(row['success']) for row in rows], dtype=np.int8)


def exact_mcnemar_p(a_only, b_only):
    n = a_only + b_only
    if not n:
        return 1.0
    lower = min(a_only, b_only)
    return min(1.0, 2 * sum(math.comb(n, k) for k in range(lower + 1)) / (2 ** n))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--relevant', type=Path, required=True)
    parser.add_argument('--static-initial', type=Path, required=True)
    parser.add_argument('--current-frame', type=Path, required=True)
    parser.add_argument('--same-task', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    paths = dict(zip(MODES, (args.relevant, args.static_initial,
                             args.current_frame, args.same_task), strict=True))
    runs = {mode: load_run(paths[mode], mode) for mode in MODES}
    reference = runs['relevant'][0]
    for mode, (report, _) in runs.items():
        for field in ('checkpoint', 'memory_adapter', 'backbone_sha256', 'plan_sha256',
                      'wait_steps', 'replan_steps', 'num_steps', 'max_steps'):
            if report[field] != reference[field]:
                raise ValueError(f'{mode} differs in {field}')
    recent = runs['relevant'][1]
    rng = np.random.default_rng(20260928)
    comparison = {}
    for mode in MODES[1:]:
        control = runs[mode][1]
        differences = recent - control
        boot = differences[rng.integers(0, 50, size=(20000, 50))].mean(axis=1)
        recent_only = int(((recent == 1) & (control == 0)).sum())
        control_only = int(((recent == 0) & (control == 1)).sum())
        comparison[mode] = dict(
            recent_only=recent_only, control_only=control_only,
            both_success=int(((recent == 1) & (control == 1)).sum()),
            both_fail=int(((recent == 0) & (control == 0)).sum()),
            gain=float(differences.mean()),
            paired_bootstrap_ci95=[float(value) for value in np.quantile(boot, [.025, .975])],
            exact_mcnemar_p=exact_mcnemar_p(recent_only, control_only),
        )
    result = dict(
        status='complete', task_id=8, trials=50,
        mode_successes={mode: int(rows.sum()) for mode, (_, rows) in runs.items()},
        comparisons_vs_relevant=comparison,
        mechanistic_caveat=('The current adapter rebuilds a fast-weight state from a recent frame '
                            'at every replan; it does not accumulate persistent online memory.'),
        sources={mode: str(path) for mode, path in paths.items()},
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
