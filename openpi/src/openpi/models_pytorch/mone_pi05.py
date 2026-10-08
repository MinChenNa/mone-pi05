"""MoNe-style test-time memory adapter for pi0.5.

This module is intentionally independent of the pi0.5 checkpoint.  It turns
historical prefix embeddings into a fixed number of memory tokens which can be
prepended to the current pi0.5 prefix.  The backbone remains frozen.

The implementation follows MoNe's two-phase contract (segment-wise write,
query-only read), while using a stable delta-rule fast-weight matrix for the
first runnable prototype.  Replacing ``DeltaFastWeightMemory`` with the
paper's meta-trained per-layer SwiGLU memory is the next fidelity step.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F  # noqa: N812


@dataclass
class MoNeState:
    """Per-batch fast weights; constant-size with respect to history length."""

    weights: Tensor
    tokens_seen: int = 0

    def detach(self) -> "MoNeState":
        return MoNeState(self.weights.detach(), self.tokens_seen)


class DeltaFastWeightMemory(nn.Module):
    """Segment-wise associative memory with a normalized delta update.

    Keys and values are split into heads.  For every token, the update writes
    the residual between the desired value and the current memory prediction.
    This is a stable fast-weight analogue of MoNe's local associative loss.
    """

    def __init__(self, embed_dim: int, num_heads: int = 4, learning_rate: float = 1.0):
        super().__init__()
        if embed_dim % num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.learning_rate = learning_rate
        self.key_norm = nn.RMSNorm(self.head_dim)
        self.output_norm = nn.RMSNorm(self.head_dim)
        self.output_gate = nn.Parameter(torch.zeros(num_heads))

    def empty_state(self, batch_size: int, *, device: torch.device, dtype: torch.dtype) -> MoNeState:
        shape = (batch_size, self.num_heads, self.head_dim, self.head_dim)
        return MoNeState(torch.zeros(shape, device=device, dtype=dtype))

    def _heads(self, x: Tensor) -> Tensor:
        return x.reshape(x.shape[0], x.shape[1], self.num_heads, self.head_dim).transpose(1, 2)

    def write_segment(self, state: MoNeState, keys: Tensor, values: Tensor, mask: Tensor | None = None) -> MoNeState:
        """Write one ``[batch, segment, embed]`` segment into fast weights."""
        if keys.shape != values.shape or keys.ndim != 3 or keys.shape[-1] != self.embed_dim:
            raise ValueError("keys and values must have shape [batch, segment, embed_dim]")
        key_heads = F.normalize(F.silu(self.key_norm(self._heads(keys))), dim=-1)
        value_heads = F.silu(self._heads(values))
        weights = state.weights
        for index in range(keys.shape[1]):
            key = key_heads[:, :, index]
            value = value_heads[:, :, index]
            prediction = torch.einsum("bhde,bhe->bhd", weights, key)
            residual = value - prediction
            update = torch.einsum("bhd,bhe->bhde", residual, key)
            if mask is not None:
                update = update * mask[:, index, None, None, None].to(update.dtype)
            weights = weights + self.learning_rate * update
        return MoNeState(weights, state.tokens_seen + keys.shape[1])

    def read(self, state: MoNeState, queries: Tensor) -> Tensor:
        """Read fixed-count memory tokens from query embeddings only."""
        query_heads = F.normalize(F.silu(self.key_norm(self._heads(queries))), dim=-1)
        output = torch.einsum("bhde,bhqe->bhqd", state.weights, query_heads)
        output = self.output_norm(output)
        # Start as an identity-safe plugin; training learns how strongly each head contributes.
        output = output * (1.0 + torch.tanh(self.output_gate)[None, :, None, None])
        return output.transpose(1, 2).reshape(queries.shape)


class Pi05MoNeContextAdapter(nn.Module):
    """Build and inject a reusable MoNe state around pi0.5 prefix embeddings."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        segment_size: int = 512,
        *,
        memory_dim: int | None = None,
        learnable_w0: bool = False,
    ):
        super().__init__()
        if segment_size <= 0:
            raise ValueError("segment_size must be positive")
        self.segment_size = segment_size
        self.embed_dim = embed_dim
        self.memory_dim = memory_dim or embed_dim
        self.memory = DeltaFastWeightMemory(self.memory_dim, num_heads)
        self.key_proj = nn.Linear(embed_dim, self.memory_dim, bias=False)
        self.value_proj = nn.Linear(embed_dim, self.memory_dim, bias=False)
        self.query_proj = nn.Linear(embed_dim, self.memory_dim, bias=False)
        self.output_proj = nn.Linear(self.memory_dim, embed_dim, bias=False)
        self.injection_gate = nn.Parameter(torch.tensor(-2.0))
        if learnable_w0:
            self.initial_weights = nn.Parameter(
                torch.zeros(num_heads, self.memory.head_dim, self.memory.head_dim)
            )
        else:
            self.register_parameter("initial_weights", None)

    def empty_state(self, batch_size: int, *, device: torch.device, dtype: torch.dtype) -> MoNeState:
        if self.initial_weights is None:
            return self.memory.empty_state(batch_size, device=device, dtype=dtype)
        weights = self.initial_weights.to(device=device, dtype=dtype)[None].expand(batch_size, -1, -1, -1)
        return MoNeState(weights)

    def build_state(self, history_embs: Tensor, history_mask: Tensor | None = None) -> MoNeState:
        state = self.empty_state(
            history_embs.shape[0], device=history_embs.device, dtype=history_embs.dtype
        )
        for start in range(0, history_embs.shape[1], self.segment_size):
            stop = start + self.segment_size
            segment = history_embs[:, start:stop]
            mask = None if history_mask is None else history_mask[:, start:stop]
            state = self.memory.write_segment(state, self.key_proj(segment), self.value_proj(segment), mask)
        return state

    def memory_tokens(self, state: MoNeState, query_embs: Tensor) -> Tensor:
        memory = self.memory.read(state, self.query_proj(query_embs))
        return self.output_proj(memory) * torch.sigmoid(self.injection_gate)

    def augment_prefix(
        self,
        prefix_embs: Tensor,
        prefix_pad_masks: Tensor,
        prefix_att_masks: Tensor,
        state: MoNeState,
        query_embs: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Prepend query-conditioned memory tokens before pi0.5 KV caching."""
        memory = self.memory_tokens(state, query_embs)
        batch_size, memory_len = memory.shape[:2]
        valid = torch.ones(batch_size, memory_len, dtype=torch.bool, device=memory.device)
        shared_block = torch.zeros(batch_size, memory_len, dtype=prefix_att_masks.dtype, device=memory.device)
        return (
            torch.cat([memory, prefix_embs], dim=1),
            torch.cat([valid, prefix_pad_masks], dim=1),
            torch.cat([shared_block, prefix_att_masks], dim=1),
        )
