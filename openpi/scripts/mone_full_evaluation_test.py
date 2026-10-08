import json
import ast
from pathlib import Path
import re

import pytest

from .check_mone_full_eval import summarize as summarize_offline
from .check_mone_full_conclusion import conclude
from .check_mone_full_rollout import summarize as summarize_rollout
from .eval_mone_full_rollout import MODES as ROLLOUT_MODES
from .plan_mone_full import build_plan, digest


def test_frozen_plan_balances_libero_suites(tmp_path):
    path = Path(__file__).resolve().parents[2] / 'vendor/libero/libero/libero/benchmark/libero_suite_task_map.py'
    mapping = ast.literal_eval(ast.parse(path.read_text()).body[0].value)
    metadata = []
    for suite in ('libero_spatial', 'libero_object', 'libero_goal', 'libero_10'):
        for name in mapping[suite]:
            task = re.sub(r'^.*?_SCENE\d+_', '', name).replace('_', ' ')
            for _ in range(3):
                metadata.append(dict(episode_index=len(metadata), tasks=[task], length=60))
    (tmp_path / 'meta').mkdir()
    (tmp_path / 'meta/episodes.jsonl').write_text('\n'.join(map(json.dumps, metadata)))
    plan = build_plan(tmp_path)
    stored = plan.pop('plan_sha256')
    assert digest(plan) == stored
    for partition, expected in (('train', 6), ('val', 2), ('test', 2)):
        for suite in ('libero_spatial', 'libero_object', 'libero_goal', 'libero_10'):
            assert sum(plan['task_lookup'][task]['suite'] == suite
                       for task in plan['partitions'][partition]) == expected
    assert len(set().union(*map(set, plan['partitions'].values()))) == 40


def test_offline_summary_rejects_missing_pairs():
    tasks = [f'task{i}' for i in range(8)]
    plan = dict(partitions=dict(test=tasks))
    reports = [dict(tasks=tasks[i:i+2], frames=1, draws=1) for i in range(0, 8, 2)]
    rows = []
    for task in tasks:
        for episode in (0, 1):
            rows.append(dict(task=task, episode=episode, frame=10, draw=0,
                             losses=dict(baseline=1.1, relevant=1., same_task=1.02,
                                         fixed_history=1.03, current_frame=1.04,
                                         parameter_control=1.05)))
    result = summarize_offline(plan, reports, rows, 'test')
    assert result['verdict'] == 'offline_selective_signal'
    with pytest.raises(ValueError, match='Incomplete'):
        summarize_offline(plan, reports, rows[:-1], 'test')


def test_rollout_summary_pairs_by_initial_state():
    tasks = [f'task{i}' for i in range(8)]
    plan = dict(partitions=dict(test=tasks))
    reports = [dict(tasks=tasks[i:i+2], trials=2) for i in range(0, 8, 2)]
    rows = [dict(task=task, trial=trial, mode=mode,
                 success=(mode == 'relevant' or (mode == 'baseline' and trial == 0)))
            for task in tasks for trial in range(2) for mode in ROLLOUT_MODES]
    result = summarize_rollout(plan, reports, rows)
    assert result['verdict'] == 'success_signal'
    assert result['paired_trials'] == 16
    with pytest.raises(ValueError, match='Incomplete'):
        summarize_rollout(plan, reports, rows[:-1])


def test_conclusion_requires_both_gates_and_matching_checkpoint(tmp_path):
    checkpoint = tmp_path / 'official'
    checkpoint.mkdir()
    baseline = dict(status='complete', checkpoint=str(checkpoint), successes=3, trials=15)
    offline = dict(status='complete', split='test', tasks=8,
                   verdict='offline_selective_signal')
    rollout = dict(status='complete', task_count=8, paired_trials=80,
                   verdict='success_signal')
    provenance = dict(checkpoint=str(checkpoint), backbone_sha256='abc',
                      memory_adapter='memory.pt', control_adapter='control.pt')
    result = conclude(baseline, offline, rollout, provenance, provenance)
    assert result['verdict'] == 'supports_history_specific_task_benefit'
    rollout['verdict'] = 'inconclusive'
    assert conclude(baseline, offline, rollout, provenance, provenance)['verdict'] == 'inconclusive'
    with pytest.raises(ValueError, match='provenance differ'):
        conclude(baseline, offline, rollout, dict(provenance, checkpoint='other'), provenance)
