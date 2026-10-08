from .eval_mone_rollout import MODES, quat_to_axisangle, summarize


def test_paired_success_summary_counts_actual_trial_pairs():
    rows = []
    for trial, outcomes in enumerate([(0, 1, 0), (1, 0, 1)]):
        for mode, success in zip(MODES, outcomes, strict=True):
            rows.append(dict(suite='libero_spatial', task_id=2, trial=trial,
                             mode=mode, success=success))
    result = summarize(rows)
    assert result['paired_trials'] == 2
    assert result['success_rate']['relevant'] == .5
    assert result['paired']['baseline']['relevant_only'] == 1
    assert result['paired']['baseline']['control_only'] == 1
    assert result['paired']['fixed_history']['ties'] == 0


def test_identity_quaternion_has_zero_axis_angle():
    assert (quat_to_axisangle([0., 0., 0., 1.]) == 0).all()
