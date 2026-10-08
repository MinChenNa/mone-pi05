"""Summarize guarded replay of the three already observed regressions."""

import argparse
import json
from pathlib import Path


def summarize(reports):
    if len(reports) != 3 or any(row.get('status') != 'complete' for row in reports):
        raise ValueError('Need all three completed replay reports')
    if len({row['case']['suite'] for row in reports}) != 3:
        raise ValueError('Duplicate or missing suites')
    if len({(row['backbone_sha256'], row['memory_adapter']) for row in reports}) != 1:
        raise ValueError('Replay provenance differs across suites')
    radii = set(row['radius'] for row in reports[0]['results'] if row['radius'] is not None)
    if any(set(item['radius'] for item in row['results'] if item['radius'] is not None) != radii
           for row in reports):
        raise ValueError('Trust radius grid differs across suites')
    by_radius = {}
    for radius in sorted(radii):
        results = [next(item for item in row['results'] if item['radius'] == radius)
                   for row in reports]
        by_radius[str(radius)] = dict(
            reproduced_successes=sum(item['success'] for item in results),
            cases=len(results),
            mean_guard_scale=sum(item['mean_guard_scale'] for item in results)/len(results),
            per_case=[dict(suite=row['case']['suite'], trial=row['case']['trial'],
                           success=item['success'], steps=item['steps'],
                           mean_guard_scale=item['mean_guard_scale'])
                      for row, item in zip(reports, results, strict=True)],
        )
    candidates = [float(radius) for radius, row in by_radius.items()
                  if float(radius) > 0 and row['reproduced_successes'] == 3]
    return dict(status='complete', diagnostic_only=True,
                reproduced_original_failures=3,
                guarded_results=by_radius,
                candidate_radii=candidates,
                verdict=('candidate_mitigates_known_failures' if candidates else
                         'no_tested_guard_mitigates_all_known_failures'),
                note=('This grid was evaluated post hoc on known failures. It does not '
                      'establish overall benefit or safety. Choose settings on separate '
                      'validation initial states, then test on fresh initial states.'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, nargs=3, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    reports = [json.loads(path.read_text()) for path in args.inputs]
    result = summarize(reports)
    result['sources'] = [str(path) for path in args.inputs]
    args.output.write_text(json.dumps(result, indent=2))
    print(f"[replay] verdict={result['verdict']} candidate_radii={result['candidate_radii']} "
          f"report={args.output}")


if __name__ == '__main__':
    main()
