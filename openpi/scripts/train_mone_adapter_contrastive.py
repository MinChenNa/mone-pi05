"""Train only the small MoNe-style adapter while keeping pi0.5 frozen."""

import argparse
import dataclasses
import io
import json
import pathlib
import random

import jax
import numpy as np
from PIL import Image
import pyarrow.parquet as pq
import safetensors.torch
import torch
import torch.nn.functional as F  # noqa: N812

from openpi import transforms
from openpi.models import model as model_api
from openpi.models_pytorch.mone_pi05 import Pi05MoNeContextAdapter
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from openpi.policies import libero_policy
from openpi.policies import policy as policy_api
from openpi.shared import normalize
from openpi.training import config as training_config


def decode_image(cell: dict) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(cell["bytes"])).convert("RGB"))


def raw_observation(row: dict, prompt: str) -> dict:
    return {
        "observation/state": np.asarray(row["state"], dtype=np.float32),
        "observation/image": decode_image(row["image"]),
        "observation/wrist_image": decode_image(row["wrist_image"]),
        "prompt": prompt,
    }


def make_policy(checkpoint: pathlib.Path, device: torch.device):
    train_cfg = training_config.get_config("pi05_libero")
    model_cfg = dataclasses.replace(train_cfg.model, pytorch_compile_mode=None)
    model = PI0Pytorch(model_cfg)
    safetensors.torch.load_model(model, str(checkpoint / "model.safetensors"))
    model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    data_config = training_config.DataConfig(
        data_transforms=transforms.Group(
            inputs=[libero_policy.LiberoInputs(model_type=model_cfg.model_type)],
            outputs=[libero_policy.LiberoOutputs()],
        ),
        model_transforms=training_config.ModelTransformFactory()(model_cfg),
        use_quantile_norm=True,
    )
    norm_stats = normalize.load(checkpoint / "assets" / "physical-intelligence" / "libero")
    policy = policy_api.Policy(
        model,
        transforms=[
            transforms.InjectDefaultPrompt(None),
            *data_config.data_transforms.inputs,
            transforms.Normalize(norm_stats, use_quantiles=True),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[],
        is_pytorch=True,
        pytorch_device="cuda",
    )
    return model, policy


@torch.no_grad()
def prefix_features(model, policy, raw: dict, device: torch.device, history_tokens: int):
    inputs = policy._input_transform(raw)  # noqa: SLF001
    observation = model_api.Observation.from_dict(
        jax.tree.map(lambda x: torch.from_numpy(np.asarray(x)).to(device)[None, ...], inputs)
    )
    images, image_masks, language, language_masks, _ = model._preprocess_observation(  # noqa: SLF001
        observation, train=False
    )
    embeddings, mask, _ = model.embed_prefix(images, image_masks, language, language_masks)
    valid = embeddings[0, mask[0]]
    if len(valid) > history_tokens:
        indices = torch.linspace(0, len(valid) - 1, history_tokens, device=device).round().long()
        valid = valid[indices]
    query = model._last_valid_tokens(embeddings, mask, 8)[0]  # noqa: SLF001
    return valid.float().cpu(), query.float().cpu()


def score(adapter, history: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
    state = adapter.build_state(history)
    memory = adapter.memory_tokens(state, query)
    return F.cosine_similarity(memory, query, dim=-1).mean(dim=-1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--data-dir", type=pathlib.Path, required=True)
    parser.add_argument("--episodes", default="0,1,2,10")
    parser.add_argument("--frames-per-episode", type=int, default=4)
    parser.add_argument("--history-tokens", type=int, default=64)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--report", type=pathlib.Path, required=True)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    model, policy = make_policy(args.checkpoint, device)
    episode_ids = [int(value) for value in args.episodes.split(",")]
    metadata = [
        json.loads(line)
        for line in (args.data_dir / "meta" / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    prompts = {item["episode_index"]: item["tasks"][0] for item in metadata}
    examples = []
    fractions = np.linspace(0.2, 0.8, args.frames_per_episode)
    for episode_id in episode_ids:
        rows = pq.read_table(
            args.data_dir / "data" / "chunk-000" / f"episode_{episode_id:06d}.parquet"
        ).to_pylist()
        for fraction in fractions:
            current_index = min(max(10, int(len(rows) * fraction)), len(rows) - 1)
            history, _ = prefix_features(
                model,
                policy,
                raw_observation(rows[current_index - 10], prompts[episode_id]),
                device,
                args.history_tokens,
            )
            _, query = prefix_features(
                model,
                policy,
                raw_observation(rows[current_index], prompts[episode_id]),
                device,
                args.history_tokens,
            )
            examples.append({"episode": episode_id, "history": history, "query": query})
            print(f"cached features {len(examples)}/{len(episode_ids) * len(fractions)}", flush=True)

    del model
    torch.cuda.empty_cache()
    adapter = Pi05MoNeContextAdapter(
        embed_dim=2048, memory_dim=256, num_heads=4, segment_size=64, learnable_w0=True
    ).to(device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    losses = []
    for step in range(args.steps):
        chosen = random.sample(examples, min(args.batch_size, len(examples)))
        histories = torch.stack([item["history"] for item in chosen]).to(device)
        queries = torch.stack([item["query"] for item in chosen]).to(device)
        negative_items = []
        for item in chosen:
            candidates = [other for other in examples if other["episode"] != item["episode"]]
            negative_items.append(random.choice(candidates))
        negative_histories = torch.stack([item["history"] for item in negative_items]).to(device)
        positive = score(adapter, histories, queries)
        negative = score(adapter, negative_histories, queries)
        ranking_loss = F.softplus(0.15 - positive + negative).mean()
        magnitude_loss = 1e-4 * adapter.memory_tokens(
            adapter.build_state(histories), queries
        ).square().mean()
        loss = ranking_loss + magnitude_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if (step + 1) % 20 == 0 or step == 0:
            accuracy = float((positive > negative).float().mean().detach().cpu())
            print(f"step {step + 1}/{args.steps} loss={losses[-1]:.5f} pair_acc={accuracy:.2f}", flush=True)

    adapter.eval()
    with torch.no_grad():
        histories = torch.stack([item["history"] for item in examples]).to(device)
        queries = torch.stack([item["query"] for item in examples]).to(device)
        negative_histories = torch.stack(
            [next(other["history"] for other in examples if other["episode"] != item["episode"]) for item in examples]
        ).to(device)
        positive = score(adapter, histories, queries)
        negative = score(adapter, negative_histories, queries)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    safetensors.torch.save_model(adapter, str(args.output))
    report = {
        "type": "bounded contrastive proxy training",
        "backbone": "official pi05_libero, frozen",
        "episodes": episode_ids,
        "samples": len(examples),
        "steps": args.steps,
        "memory_dim": 256,
        "history_tokens": args.history_tokens,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "training_pair_accuracy_percent": float((positive > negative).float().mean().cpu() * 100),
        "mean_positive_cosine": float(positive.mean().cpu()),
        "mean_negative_cosine": float(negative.mean().cpu()),
        "limitation": "Contrastive proxy training; action quality must be evaluated separately on held-out episodes.",
    }
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
