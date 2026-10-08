"""Inference-only trust region around the frozen pi0.5 action chunk.

The guard does not train or change the MoNe adapter. It limits its proposed
change in normalized LIBERO action coordinates before unnormalization.
"""

import torch


def cap_action_delta(baseline: torch.Tensor, proposal: torch.Tensor, radius: float,
                     action_dim: int = 7) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return guarded actions, per-batch scale, and unguarded maximum delta.

    The first ``action_dim`` coordinates are used by LIBERO. Padded coordinates
    are copied from the baseline so a bound on the executed action is explicit.
    """
    if radius < 0 or not torch.isfinite(torch.tensor(radius)):
        raise ValueError('radius must be finite and nonnegative')
    if baseline.shape != proposal.shape or baseline.ndim != 3:
        raise ValueError('Expected equal [batch, horizon, action_dim] chunks')
    if not 0 < action_dim <= baseline.shape[-1]:
        raise ValueError('Invalid executed action dimension')
    if not torch.isfinite(baseline).all() or not torch.isfinite(proposal).all():
        raise ValueError('Nonfinite action chunk')
    delta = proposal[..., :action_dim] - baseline[..., :action_dim]
    maximum = delta.abs().amax(dim=(-2, -1))
    scale = (radius / maximum.clamp_min(1e-12)).clamp(max=1.0)
    guarded = baseline.clone()
    guarded[..., :action_dim] = baseline[..., :action_dim] + scale[:, None, None] * delta
    return guarded, scale, maximum
