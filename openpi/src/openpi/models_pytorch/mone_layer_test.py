import torch
from openpi.models_pytorch.mone_layer import LayerMemory


def test_empty_memory_is_identity_and_history_receives_gradients():
    torch.manual_seed(4)
    adapter = LayerMemory(embed_dim=16, memory_dim=8, segment_size=4)
    query = torch.randn(2, 5, 16)
    empty = adapter.empty_state(2, device=query.device, dtype=query.dtype)
    torch.testing.assert_close(adapter.inject(query, empty), query, rtol=0, atol=0)
    state = adapter.build_state(torch.randn(2, 8, 16))
    result = adapter.inject(query, state)
    assert not torch.equal(result, query)
    result.square().mean().backward()
    for name in ('key_proj', 'value_proj', 'query_proj', 'output_proj'):
        grad = getattr(adapter, name).weight.grad
        assert grad is not None and torch.isfinite(grad).all() and grad.norm() > 0


def test_padding_cannot_write_memory():
    adapter = LayerMemory(embed_dim=16, memory_dim=8)
    history = torch.randn(2, 4, 16)
    state = adapter.build_state(history, torch.zeros(2, 4, dtype=torch.bool))
    assert torch.count_nonzero(state.weights) == 0
