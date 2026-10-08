"""Combine preregistered offline and closed-loop gates without overclaiming."""

import argparse
import json
from pathlib import Path


def conclude(baseline, offline, rollout, offline_source, rollout_source):
    if any(report.get('status') != 'complete' for report in (baseline, offline, rollout)):
        raise ValueError('All three reports must be complete')
    if offline.get('split') != 'test' or offline.get('tasks') != 8:
        raise ValueError('Only frozen eight-task test results can support a conclusion')
    if rollout.get('task_count') != 8 or rollout.get('paired_trials', 0) < 80:
        raise ValueError('At least 80 paired test initial states are required')
    if baseline.get('successes', 0) <= 0:
        raise ValueError('The official pi0.5 competence gate did not pass')
    for key in ('backbone_sha256', 'memory_adapter', 'control_adapter', 'checkpoint'):
        if offline_source.get(key) != rollout_source.get(key):
            raise ValueError(f'Offline and rollout provenance differ: {key}')
    if (Path(baseline['checkpoint']).resolve() !=
            Path(offline_source['checkpoint']).resolve()):
        raise ValueError('Official baseline used a different backbone')
    offline_pass = offline.get('verdict') == 'offline_selective_signal'
    rollout_pass = rollout.get('verdict') == 'success_signal'
    return dict(
        status='complete',
        verdict=('supports_history_specific_task_benefit' if offline_pass and rollout_pass
                 else 'inconclusive'),
        gates=dict(official_baseline=True, offline_selectivity=offline_pass,
                   paired_closed_loop=rollout_pass),
        official_baseline=dict(successes=baseline['successes'], trials=baseline['trials']),
        test_tasks=8, paired_rollout_initial_states=rollout['paired_trials'],
        backbone_sha256=offline_source['backbone_sha256'],
        memory_adapter=offline_source['memory_adapter'],
        parameter_control=offline_source['control_adapter'],
        note=('A positive verdict supports this implementation on held-out LIBERO tasks, '
              'not the general idea across environments. An inconclusive result is not '
              'evidence that the idea is invalid; inspect effect sizes, CI and ceiling effects.'),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--offline', type=Path, required=True)
    parser.add_argument('--rollout', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    baseline = json.loads(args.baseline.read_text())
    offline = json.loads(args.offline.read_text())
    rollout = json.loads(args.rollout.read_text())
    if not offline.get('sources') or not rollout.get('sources'):
        raise ValueError('Missing shard provenance')
    offline_source = json.loads(Path(offline['sources'][0]).read_text())
    rollout_source = json.loads(Path(rollout['sources'][0]).read_text())
    result = conclude(baseline, offline, rollout, offline_source, rollout_source)
    result['reports'] = dict(baseline=str(args.baseline), offline=str(args.offline),
                             rollout=str(args.rollout))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(f"[conclusion] verdict={result['verdict']} report={args.output}")


if __name__ == '__main__':
    main()
