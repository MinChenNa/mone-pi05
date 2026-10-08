import numpy as np
import pytest

from .check_mone_history_screen import summarize
from .eval_mone_history_screen import CONDITIONS, mask_current


def test_masks_only_current_images_and_preserve_raw_input():
    raw = {'observation/image': np.full((8, 8, 3), 42, np.uint8),
           'observation/wrist_image': np.full((8, 8, 3), 42, np.uint8),
           'observation/state': np.arange(8, dtype=np.float32), 'prompt': 'task'}
    wrist = mask_current(raw, 'no_wrist')
    center = mask_current(raw, 'center_occluded')
    assert np.all(wrist['observation/wrist_image'] == 127)
    assert np.all(wrist['observation/image'] == 42)
    assert np.any(center['observation/image'] == 127)
    assert np.any(center['observation/image'] == 42)
    assert np.all(raw['observation/image'] == 42)
    assert np.all(raw['observation/wrist_image'] == 42)
    assert center['observation/state'] is raw['observation/state']


def test_summary_checks_complete_pairs_and_task_direction():
    cases = [dict(split='screen', task=f'task{i}', episode=i, frames=[10]) for i in range(12)]
    plan = dict(cases=cases, settings=dict(draws=1), plan_sha256='hash')
    records = []
    for case in cases:
        for condition in CONDITIONS:
            relevant = 1. if condition == 'clean' else 1.01
            losses = dict(baseline=1.01 if condition == 'clean' else 1.03,
                          relevant=relevant, same_task=relevant+.005,
                          fixed_history=relevant+.006, current_frame=relevant+.007)
            records.append(dict(task=case['task'], episode=case['episode'], frame=10,
                                condition=condition, draw=0, losses=losses))
    result = summarize(plan, records, 'screen')
    assert result['samples'] == 36
    assert result['verdict'] == 'screen_positive'
    assert result['effects']['center_occluded']['same_task']['task_wins'] == 12
    with pytest.raises(ValueError, match='Missing'):
        summarize(plan, records[:-1], 'screen')
