import pytest
import torch

from openpi.models_pytorch.mone_guard import cap_action_delta


def test_guard_bounds_executed_actions_and_preserves_padding():
    baseline = torch.zeros(2, 5, 32)
    proposal = torch.ones_like(baseline)
    guarded, scale, maximum = cap_action_delta(baseline, proposal, .05)
    torch.testing.assert_close(guarded[..., :7], torch.full((2, 5, 7), .05))
    torch.testing.assert_close(guarded[..., 7:], baseline[..., 7:])
    torch.testing.assert_close(scale, torch.full((2,), .05))
    torch.testing.assert_close(maximum, torch.ones(2))


def test_guard_zero_is_exact_baseline_and_large_radius_is_proposal():
    baseline = torch.randn(1, 10, 32)
    proposal = torch.randn_like(baseline)
    zero, _, _ = cap_action_delta(baseline, proposal, 0)
    assert torch.equal(zero, baseline)
    wide, _, _ = cap_action_delta(baseline, proposal, 100)
    torch.testing.assert_close(wide[..., :7], proposal[..., :7])
    assert torch.equal(wide[..., 7:], baseline[..., 7:])


def test_guard_rejects_bad_inputs():
    with pytest.raises(ValueError, match='nonnegative'):
        cap_action_delta(torch.zeros(1, 5, 32), torch.zeros(1, 5, 32), -1)
    with pytest.raises(ValueError, match='equal'):
        cap_action_delta(torch.zeros(1, 5, 32), torch.zeros(1, 4, 32), .1)
