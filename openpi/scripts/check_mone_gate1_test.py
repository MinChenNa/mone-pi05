from .check_mone_gate1 import assess


def report(advantages, baseline=.35):
    return dict(status='complete', cases=[dict(episode=i) for i in range(len(advantages))],
                summary_action7_task=dict(per_task={str(i): dict(
                    baseline=baseline, relevant=baseline-advantages[i],
                    wrong_task=baseline-advantages[i]+.003,
                    same_task=baseline-advantages[i]+.003,
                    current_frame=baseline-advantages[i]+.003,
                    fixed_history=baseline-advantages[i]+.003)
                    for i in range(len(advantages))}))


def test_gate_requires_paired_control_advantage_and_quality():
    old = report([.001]*12)
    new = report([.004]*12)
    train = dict(status='complete', tasks=['other task'])
    assert assess(old, new, train, .001)['verdict'] == 'pass_offline_selectivity'
    for task in new['summary_action7_task']['per_task'].values():
        task['fixed_history'] = task['relevant'] - .001
    assert assess(old, new, train, .001)['verdict'] == 'fail_current_adapter'
