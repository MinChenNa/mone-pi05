import torch

from .mone_selective_objective import selective_objective


def test_negative_branch_is_capped_at_backbone_loss():
    positive = torch.tensor(.3, requires_grad=True)
    negative = torch.tensor(.4, requires_grad=True)
    total, _ = selective_objective(positive, negative, torch.tensor(.35))
    total.backward()
    assert positive.grad > 0
    assert negative.grad == 0


def test_ranking_pushes_uncapped_wrong_history_up():
    positive = torch.tensor(.3, requires_grad=True)
    negative = torch.tensor(.31, requires_grad=True)
    total, _ = selective_objective(positive, negative, torch.tensor(.35))
    total.backward()
    assert positive.grad > 0
    assert negative.grad < 0
