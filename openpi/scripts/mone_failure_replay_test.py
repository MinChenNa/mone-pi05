import json
from pathlib import Path

from .check_mone_failure_replay import summarize
from .eval_mone_failure_replay import regression_cases


def test_frozen_rollout_has_exact_regression_cases():
    report_path = (Path(__file__).resolve().parents[2] /
                   'outputs/full18/full18_20260927/rollout-paired.json')
    report = json.loads(report_path.read_text())
    # Sources in the frozen report are relative to the openpi working directory.
    import os
    previous = Path.cwd()
    try:
        os.chdir(Path(__file__).resolve().parents[1])
        cases = regression_cases(report)
    finally:
        os.chdir(previous)
    assert [(item['suite'], item['task_id'], item['trial']) for item in cases] == [
        ('libero_goal', 3, 6), ('libero_object', 5, 1), ('libero_spatial', 9, 1)]


def test_replay_summary_marks_posthoc_candidate():
    reports = []
    for suite in ('libero_goal', 'libero_object', 'libero_spatial'):
        reports.append(dict(status='complete', case=dict(suite=suite, trial=1),
                            backbone_sha256='abc', memory_adapter='adapter.pt',
                            results=[dict(radius=None, mode='baseline', success=True),
                                     dict(radius=None, mode='relevant', success=False),
                                     dict(radius=0, mode='guarded_0', success=True,
                                          steps=100, mean_guard_scale=0),
                                     dict(radius=.05, mode='guarded_0p05', success=True,
                                          steps=110, mean_guard_scale=.4)]))
    result = summarize(reports)
    assert result['candidate_radii'] == [.05]
    assert result['diagnostic_only']
