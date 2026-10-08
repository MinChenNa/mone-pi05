"""Independent MoNe blocks at all 18 pi0.5 context Transformer layers.

Historical prefix features are captured in one frozen forward pass. Each
context layer writes its own fixed-size memory and reads it only through
that layer's normalized attention input.
"""

from contextlib import contextmanager

import torch
from torch import nn
import torch.nn.functional as F

from openpi.models_pytorch.mone_layer import LayerMemory


@torch.no_grad()
def capture_context_layers(model, observation, max_tokens=32):
    """Return [batch, layer, sampled_token, width] and its padding mask."""
    from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks
    if max_tokens < 1:
        raise ValueError('max_tokens must be positive')
    layers = model.paligemma_with_expert.paligemma.language_model.layers
    images, image_masks, tokens, token_masks, _ = model._preprocess_observation(
        observation, train=False)
    prefix, valid, blocks = model.embed_prefix(images, image_masks, tokens, token_masks)
    indices = torch.linspace(0, prefix.shape[1]-1, min(max_tokens, prefix.shape[1]),
                             device=prefix.device).round().long()
    captured = [None] * len(layers)
    handles = []
    for layer_index, layer in enumerate(layers):
        def hook(module, args, output, index=layer_index):
            captured[index] = output[0][:, indices].detach().float()
        handles.append(layer.input_layernorm.register_forward_hook(hook))
    try:
        attention = model._prepare_attention_masks_4d(make_att_2d_masks(valid, blocks))
        model.paligemma_with_expert(
            inputs_embeds=[prefix, None], attention_mask=attention,
            position_ids=valid.long().cumsum(-1)-1, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    if any(item is None for item in captured):
        raise RuntimeError('Not all context layers produced historical features')
    return torch.stack(captured, dim=1), valid[:, indices]


class AllLayerMemory(nn.Module):
    """One independent memory block per context layer."""

    def __init__(self, depth=18, context_dim=2048, memory_dim=256):
        super().__init__()
        if depth < 1 or memory_dim < 1:
            raise ValueError('depth and memory_dim must be positive')
        self.depth = depth
        self.context = nn.ModuleList([
            LayerMemory(context_dim, memory_dim, segment_size=32) for _ in range(depth)])

    def build_states(self, features, mask):
        if features.ndim != 4 or features.shape[1] != self.depth:
            raise ValueError(f'Expected history features [batch,{self.depth},tokens,width]')
        if mask.shape != features.shape[:1] + features.shape[2:3]:
            raise ValueError('History mask shape mismatch')
        return [block.build_state(features[:, index], mask)
                for index, block in enumerate(self.context)]

    def empty_states(self, batch_size, *, device, dtype=torch.float32):
        return [block.empty_state(batch_size, device=device, dtype=dtype)
                for block in self.context]

    @contextmanager
    def activate(self, model, states):
        context_layers = model.paligemma_with_expert.paligemma.language_model.layers
        if len(context_layers) != self.depth or len(states) != self.depth:
            raise ValueError('Context depth mismatch')
        handles = []

        def attach(layers, blocks, branch_states):
            for layer, block, state in zip(layers, blocks, branch_states, strict=True):
                def hook(module, args, output, adapter=block, memory_state=state):
                    hidden, gate = output
                    return adapter.inject(hidden, memory_state), gate
                handles.append(layer.input_layernorm.register_forward_hook(hook))

        attach(context_layers, self.context, states)
        try:
            yield
        finally:
            for handle in handles:
                handle.remove()

    def _loss_with_states(self, model, current, actions, states, noise, time):
        with self.activate(model, states):
            return model(current, actions, noise=noise, time=time,
                         preprocess_train=False)[..., :7].mean()

    def forward(self, model, current, actions, features, mask, noise, time,
                *, empty=False, negative_features=None, baseline=None,
                margin=0.002, temperature=0.01, contrast_weight=1.0):
        states = (self.empty_states(features.shape[0], device=features.device)
                  if empty else self.build_states(features, mask))
        positive = self._loss_with_states(model, current, actions, states, noise, time)
        if negative_features is None:
            return positive
        if baseline is None or temperature <= 0:
            raise ValueError('Paired training requires baseline and positive temperature')
        negative_states = self.build_states(negative_features, mask)
        negative = self._loss_with_states(model, current, actions, negative_states, noise, time)
        capped = torch.minimum(negative, baseline.detach())
        ranking = temperature * F.softplus((positive - capped + margin) / temperature)
        return positive + contrast_weight * ranking, positive.detach(), negative.detach()
