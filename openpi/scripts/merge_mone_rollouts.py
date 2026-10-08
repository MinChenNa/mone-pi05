"""Combine independently run LIBERO suites into one paired pilot report."""

import argparse
import json
from pathlib import Path

if __package__:
    from .eval_mone_rollout import summarize
else:
    from eval_mone_rollout import summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    reports = [json.loads(path.read_text()) for path in args.inputs]
    if any(report.get('status') != 'complete' for report in reports):
        raise ValueError('All rollout reports must be complete')
    records = [json.loads(line) for path in args.inputs
               for line in path.with_suffix('.jsonl').read_text().splitlines()]
    selected = [task for report in reports for task in report['selected_tasks']]
    if len({(task['suite'], task['task_id']) for task in selected}) != len(selected):
        raise ValueError('Duplicate task across rollout reports')
    if len({report['adapter'] for report in reports}) != 1:
        raise ValueError('Different adapters across rollout reports')
    merged = dict(status='complete', adapter=reports[0]['adapter'], selected_tasks=selected,
                  source_reports=[str(path) for path in args.inputs], summary=summarize(records))
    args.output.write_text(json.dumps(merged, indent=2))
    print(f'[done] {args.output} success_rate={merged["summary"]["success_rate"]}', flush=True)


if __name__ == '__main__':
    main()
