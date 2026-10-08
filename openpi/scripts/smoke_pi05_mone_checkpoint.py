"""Run baseline and MoNe-augmented inference with a real pi0.5 checkpoint."""

import argparse
import json
import time
import types

import safetensors.torch
import torch

from openpi.models import pi0_config
from openpi.models_pytorch.mone_pi05 import Pi05MoNeContextAdapter
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch


def fake_observation(device: torch.device, *, prompt_token: int):
    images = {
        key: torch.zeros(1, 3, 224, 224, dtype=torch.float32, device=device)
        for key in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
    }
    image_masks = {key: torch.ones(1, dtype=torch.bool, device=device) for key in images}
    tokens = torch.zeros(1, 200, dtype=torch.long, device=device)
    tokens[:, :8] = prompt_token
    token_mask = torch.zeros(1, 200, dtype=torch.bool, device=device)
    token_mask[:, :8] = True
    return types.SimpleNamespace(
        images=images,
        image_masks=image_masks,
        state=torch.zeros(1, 32, dtype=torch.float32, device=device),
        tokenized_prompt=tokens,
        tokenized_prompt_mask=token_mask,
        token_ar_mask=None,
        token_loss_mask=None,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-steps", type=int, default=2)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this checkpoint smoke test")
    device = torch.device("cuda")
    config = pi0_config.Pi0Config(
        pi05=True,
        action_horizon=10,
        discrete_state_input=False,
        pytorch_compile_mode=None,
    )
    model = PI0Pytorch(config)
    safetensors.torch.load_model(model, args.checkpoint)
    # Preserve openpi's intended mixed precision: transformer weights use
    # bfloat16 while action projections remain float32.
    model = model.to(device=device).eval()

    adapter = Pi05MoNeContextAdapter(embed_dim=2048, num_heads=4, segment_size=64)
    with torch.no_grad():
        identity = torch.eye(2048)
        adapter.key_proj.weight.copy_(identity)
        adapter.value_proj.weight.copy_(identity)
        adapter.query_proj.weight.copy_(identity)
    model.attach_mone(adapter.to(device=device, dtype=torch.bfloat16))

    history = fake_observation(device, prompt_token=42)
    current = fake_observation(device, prompt_token=42)
    noise = torch.randn(1, 10, 32, device=device, generator=torch.Generator(device=device).manual_seed(2026))

    torch.cuda.reset_peak_memory_stats()
    state_start = time.perf_counter()
    state = model.build_mone_state(history)
    torch.cuda.synchronize()
    state_ms = (time.perf_counter() - state_start) * 1000

    baseline_start = time.perf_counter()
    baseline = model.sample_actions(device, current, noise=noise.clone(), num_steps=args.num_steps)
    torch.cuda.synchronize()
    baseline_ms = (time.perf_counter() - baseline_start) * 1000

    mone_start = time.perf_counter()
    augmented = model.sample_actions(
        device,
        current,
        noise=noise.clone(),
        num_steps=args.num_steps,
        mone_state=state,
        memory_query_tokens=8,
    )
    torch.cuda.synchronize()
    mone_ms = (time.perf_counter() - mone_start) * 1000

    result = {
        "checkpoint": args.checkpoint,
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "num_steps": args.num_steps,
        "state_shape": list(state.weights.shape),
        "state_bytes": state.weights.numel() * state.weights.element_size(),
        "history_tokens_seen": state.tokens_seen,
        "memory_query_tokens": 8,
        "baseline_finite": bool(torch.isfinite(baseline).all()),
        "mone_finite": bool(torch.isfinite(augmented).all()),
        "action_mean_abs_delta": float((augmented - baseline).abs().mean()),
        "action_max_abs_delta": float((augmented - baseline).abs().max()),
        "state_build_ms": round(state_ms, 3),
        "baseline_ms": round(baseline_ms, 3),
        "mone_ms": round(mone_ms, 3),
        "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
    }
    with open(args.output, "w", encoding="utf-8") as output_file:
        json.dump(result, output_file, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
