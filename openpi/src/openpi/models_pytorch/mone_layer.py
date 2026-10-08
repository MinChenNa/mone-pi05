"""Single-layer residual memory prototype (not a full MoNe reproduction)."""

from contextlib import contextmanager

import torch
import torch.nn.functional as F

from openpi.models_pytorch.mone_pi05 import Pi05MoNeContextAdapter


class LayerMemory(Pi05MoNeContextAdapter):
    """Inject bounded memory into normalized PaliGemma attention inputs.

    The hook is shared by joint training and prefix-cache inference. Keep the
    context open through backward so checkpoint recomputation sees the hook.
    No tokens, positions, or attention masks are added. Not thread safe.
    """

    def __init__(self, embed_dim=2048, memory_dim=256, segment_size=64):
        super().__init__(embed_dim, memory_dim=memory_dim, segment_size=segment_size)
        self.injection_gate.data.fill_(-4.0)

    def inject(self, hidden, state):
        # Memory arithmetic remains float32 even with a bfloat16 backbone.
        query = hidden.to(self.query_proj.weight.dtype)
        memory = self.memory.read(state, self.query_proj(query))
        residual = F.rms_norm(self.output_proj(memory), (self.embed_dim,))
        return (query + torch.sigmoid(self.injection_gate) * residual).to(hidden.dtype)

    @contextmanager
    def activate(self, model, state, layer=9):
        norm = model.paligemma_with_expert.paligemma.language_model.layers[layer].input_layernorm

        def hook(module, args, output):
            hidden, gate = output
            return self.inject(hidden, state), gate

        handle = norm.register_forward_hook(hook)
        try:
            yield
        finally:
            handle.remove()


@torch.no_grad()
def capture_history(model, observation, layer=9, max_tokens=64):
    """Cache frozen normalized features at the insertion layer; stop there."""
    if max_tokens < 1:
        raise ValueError("max_tokens must be positive")
    from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks

    images, masks, tokens, token_masks, _ = model._preprocess_observation(observation, train=False)
    prefix, valid, blocks = model.embed_prefix(images, masks, tokens, token_masks)
    norm = model.paligemma_with_expert.paligemma.language_model.layers[layer].input_layernorm
    captured = []

    class Captured(Exception):
        pass

    def hook(module, args, output):
        captured.append(output[0].detach().float())
        raise Captured

    handle = norm.register_forward_hook(hook)
    try:
        attention_mask = model._prepare_attention_masks_4d(make_att_2d_masks(valid, blocks))
        model.paligemma_with_expert(
            inputs_embeds=[prefix, None],
            attention_mask=attention_mask,
            position_ids=valid.long().cumsum(-1) - 1,
            use_cache=False,
        )
    except Captured:
        pass
    finally:
        handle.remove()
    # Keep a shared token axis and the real padding mask, including mixed batches.
    indices = torch.linspace(0, prefix.shape[1] - 1, min(max_tokens, prefix.shape[1]),
                             device=prefix.device).round().long()
    return captured[0][:, indices], valid[:, indices]
