"""Action loss plus a bounded paired preference for the correct history."""

import torch
import torch.nn.functional as F


def selective_objective(positive, negative, baseline, *, margin=0.002,
                        temperature=0.01, contrast_weight=1.0):
    """Prefer the correct history without rewarding unlimited wrong-history harm.

    Negative loss can rise only until the frozen backbone baseline. Above that
    point the ranking term has no gradient through the negative branch.
    """
    if margin < 0 or temperature <= 0 or contrast_weight < 0:
        raise ValueError('margin and contrast_weight must be nonnegative; temperature positive')
    capped_negative = torch.minimum(negative, baseline.detach())
    ranking = temperature * F.softplus((positive - capped_negative + margin) / temperature)
    return positive + contrast_weight * ranking, ranking
