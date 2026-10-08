import torch
from torch import nn

from openpi.models_pytorch.mone_all_layers import AllLayerMemory


class PairNorm(nn.Module):
    def forward(self, hidden):
        return hidden, torch.ones_like(hidden)


class FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.paligemma_with_expert = nn.Module()
        self.paligemma_with_expert.paligemma = nn.Module()
        self.paligemma_with_expert.paligemma.language_model = nn.Module()
        self.paligemma_with_expert.paligemma.language_model.layers = nn.ModuleList()
        for _ in range(2):
            layer = nn.Module()
            layer.input_layernorm = PairNorm()
            self.paligemma_with_expert.paligemma.language_model.layers.append(layer)

    def forward(self, current, actions, noise, time, preprocess_train):
        hidden = current
        for layer in self.paligemma_with_expert.paligemma.language_model.layers:
            hidden, _ = layer.input_layernorm(hidden)
        return (hidden - actions).square()


def test_all_layers_use_separate_hooks_and_backpropagate():
    torch.manual_seed(1)
    model = FakeModel()
    adapter = AllLayerMemory(depth=2, context_dim=8, memory_dim=8)
    features = torch.randn(1, 2, 4, 8)
    mask = torch.ones(1, 4, dtype=torch.bool)
    current = torch.randn(1, 1, 8)
    actions = torch.zeros_like(current)
    noise = torch.zeros_like(current)
    time = torch.zeros(1)
    unmodified = model(current, actions, noise, time, False)[..., :7].mean()
    empty = adapter(model, current, actions, features, mask, noise, time, empty=True)
    torch.testing.assert_close(unmodified, empty, atol=1e-6, rtol=0)
    positive = adapter(model, current, actions, features, mask, noise, time)
    positive.backward()
    assert all(block.output_proj.weight.grad is not None for block in adapter.context)
    assert all(len(layer.input_layernorm._forward_hooks) == 0
               for layer in model.paligemma_with_expert.paligemma.language_model.layers)


def test_paired_training_returns_finite_losses():
    model = FakeModel()
    adapter = AllLayerMemory(depth=2, context_dim=8, memory_dim=8)
    current = torch.randn(1, 1, 8)
    actions = torch.zeros_like(current)
    features = torch.randn(1, 2, 3, 8)
    mask = torch.ones(1, 3, dtype=torch.bool)
    baseline = model(current, actions, current, torch.zeros(1), False)[..., :7].mean()
    total, pos, neg = adapter(model, current, actions, features, mask,
                              torch.zeros_like(current), torch.zeros(1),
                              negative_features=-features, baseline=baseline)
    assert all(torch.isfinite(value) for value in (total, pos, neg))
