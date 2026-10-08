import json

import numpy as np
import torch

from .eval_mone_heldout import MODES, TRAIN_EPISODES, plan, reverse_valid, summarize


def test_reverse_changes_valid_order_without_moving_padding():
    values = torch.arange(5).reshape(1,5,1)
    mask = torch.tensor([[True, False, True, True, False]])
    actual = reverse_valid(values, mask)
    assert actual.flatten().tolist() == [3,1,2,0,4]
    # Old roll shifts final padding to the front; valid write order is unchanged.
    assert torch.equal(values[mask], values.roll(1,1)[mask.roll(1,1)])


def test_plan_excludes_train_and_uses_disjoint_donors(tmp_path):
    (tmp_path/'meta').mkdir()
    metadata = [dict(episode_index=i, tasks=[f'task_{i%3}'], length=60) for i in range(40)]
    (tmp_path/'meta/episodes.jsonl').write_text('\n'.join(map(json.dumps,metadata)))
    cases = plan(tmp_path, 3, 123)
    assert cases == plan(tmp_path, 3, 123)
    evaluation_ids = {c['episode'] for c in cases}
    donors = {c[k] for c in cases for k in ('same_episode','wrong_episode')}
    assert not (evaluation_ids | donors) & TRAIN_EPISODES
    assert not evaluation_ids & donors
    assert all(c['task'] != c['wrong_task'] for c in cases)
    assert all(c['episode']%3 == c['same_episode']%3 for c in cases)
    excluded = {'task_1'}
    filtered = plan(tmp_path, 2, 123, excluded)
    assert all(c['task'] not in excluded and c['wrong_task'] not in excluded for c in filtered)


def test_summary_weights_episodes_equally_and_pairs_controls():
    def record(ep, baseline):
        losses = {mode:1. for mode in MODES}
        losses['baseline'] = baseline
        return dict(episode=ep, task=f'task_{ep}', losses=losses)
    result = summarize([record(1,3.)]*9 + [record(2,1.)])
    assert result['mean_loss']['baseline'] == 2.
    assert result['gain_vs_baseline']['mean'] == 1.
    assert result['gain_vs_baseline']['episode_win_fraction'] == .5
    np.testing.assert_array_equal(result['gain_vs_wrong_task']['ci95'], [0.,0.])
    assert result['gain_vs_current_frame']['mean'] == 0.
    assert result['gain_vs_fixed_history']['mean'] == 0.
    task_result = summarize([record(1,3.)]*9 + [record(2,1.)], group_key='task')
    assert task_result['task_count'] == 2
    assert task_result['gain_vs_baseline']['task_win_fraction'] == .5
