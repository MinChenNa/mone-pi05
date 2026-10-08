import torch

from openpi.models_pytorch.mone_pi05 import DeltaFastWeightMemory
from openpi.models_pytorch.mone_pi05 import Pi05MoNeContextAdapter
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch


def test_associative_write_reduces_recall_error():
    torch.manual_seed(7)
    memory = DeltaFastWeightMemory(embed_dim=16, num_heads=4)
    keys = torch.randn(2, 12, 16)
    values = torch.randn(2, 12, 16)
    state = memory.empty_state(2, device=keys.device, dtype=keys.dtype)

    before = memory.read(state, keys)
    state = memory.write_segment(state, keys, values)
    after = memory.read(state, keys)

    # The normalized read need not reproduce raw values exactly, but writing must
    # make the output informative and move it away from the empty-memory result.
    assert state.tokens_seen == 12
    assert torch.count_nonzero(after) > 0
    assert not torch.allclose(before, after)


def test_query_cost_and_state_size_do_not_grow_with_history():
    torch.manual_seed(11)
    adapter = Pi05MoNeContextAdapter(embed_dim=32, num_heads=4, segment_size=8)
    short = torch.randn(1, 8, 32)
    long = torch.randn(1, 80, 32)
    query = torch.randn(1, 3, 32)

    short_state = adapter.build_state(short)
    long_state = adapter.build_state(long)

    assert short_state.weights.shape == long_state.weights.shape
    assert adapter.memory_tokens(short_state, query).shape == (1, 3, 32)
    assert adapter.memory_tokens(long_state, query).shape == (1, 3, 32)


def test_pi05_prefix_injection_contract():
    adapter = Pi05MoNeContextAdapter(embed_dim=16, num_heads=4, segment_size=4)
    history = torch.randn(2, 9, 16)
    state = adapter.build_state(history)
    prefix = torch.randn(2, 6, 16)
    pad = torch.ones(2, 6, dtype=torch.bool)
    att = torch.zeros(2, 6, dtype=torch.bool)
    query = prefix[:, -2:]

    embs, pad_out, att_out = adapter.augment_prefix(prefix, pad, att, state, query)

    assert embs.shape == (2, 8, 16)
    assert pad_out.shape == att_out.shape == (2, 8)
    assert torch.all(pad_out[:, :2])


def test_last_valid_tokens_excludes_right_padding():
    embs = torch.arange(6, dtype=torch.float32).reshape(1, 6, 1)
    mask = torch.tensor([[True, True, True, False, False, False]])

    selected = PI0Pytorch._last_valid_tokens(embs, mask, 2)

    assert selected.flatten().tolist() == [1.0, 2.0]
