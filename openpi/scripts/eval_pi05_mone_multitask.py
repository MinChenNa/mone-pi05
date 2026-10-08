"""Multi-task offline diagnostic for an untrained MoNe-style pi0.5 adapter."""

import argparse
import dataclasses
import io
import json
import pathlib

import jax
import numpy as np
from PIL import Image
import pyarrow.parquet as pq
import safetensors.torch
from scipy import stats
import torch

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


def transformed_observation(policy, raw: dict, device: torch.device):
    inputs = policy._input_transform(raw)  # noqa: SLF001
    torch_inputs = jax.tree.map(lambda x: torch.from_numpy(np.asarray(x)).to(device)[None, ...], inputs)
    return inputs, model_api.Observation.from_dict(torch_inputs)


def unnormalize_actions(policy, transformed_inputs: dict, actions: torch.Tensor) -> np.ndarray:
    outputs = {
        "state": np.asarray(transformed_inputs["state"]),
        "actions": actions[0].detach().float().cpu().numpy(),
    }
    return np.asarray(policy._output_transform(outputs)["actions"])  # noqa: SLF001


def bootstrap_mean_ci(values: np.ndarray, rng: np.random.Generator, draws: int = 10_000) -> list[float]:
    samples = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    return [float(x) for x in np.quantile(samples, [0.025, 0.975])]


def summary(name: str, mse: np.ndarray, baseline: np.ndarray, rng: np.random.Generator) -> dict:
    improvement = baseline - mse
    wilcoxon = stats.wilcoxon(improvement, alternative="greater")
    return {
        "condition": name,
        "mean_mse": float(mse.mean()),
        "median_mse": float(np.median(mse)),
        "relative_mean_mse_change_percent": float((mse.mean() / baseline.mean() - 1.0) * 100.0),
        "win_rate_percent": float((mse < baseline).mean() * 100.0),
        "mean_absolute_mse_improvement": float(improvement.mean()),
        "improvement_bootstrap_95_ci": bootstrap_mean_ci(improvement, rng),
        "wilcoxon_one_sided_p": float(wilcoxon.pvalue),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", type=pathlib.Path, required=True)
    parser.add_argument("--episodes", default="0,1,2,10,20,30")
    parser.add_argument("--frames-per-episode", type=int, default=4)
    parser.add_argument("--num-steps", type=int, default=2)
    parser.add_argument("--adapter-checkpoint")
    parser.add_argument("--memory-dim", type=int, default=2048)
    parser.add_argument("--learnable-w0", action="store_true")
    parser.add_argument("--injection-gate", type=float)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = torch.device("cuda")
    train_cfg = training_config.get_config("pi05_libero")
    model_cfg = dataclasses.replace(train_cfg.model, pytorch_compile_mode=None)
    model = PI0Pytorch(model_cfg)
    safetensors.torch.load_model(model, str(pathlib.Path(args.checkpoint) / "model.safetensors"))
    model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    data_config = training_config.DataConfig(
        data_transforms=transforms.Group(
            inputs=[libero_policy.LiberoInputs(model_type=model_cfg.model_type)],
            outputs=[libero_policy.LiberoOutputs()],
        ),
        model_transforms=training_config.ModelTransformFactory()(model_cfg),
        use_quantile_norm=True,
    )
    norm_stats = normalize.load(
        pathlib.Path(args.checkpoint) / "assets" / "physical-intelligence" / "libero"
    )
    policy = policy_api.Policy(
        model,
        transforms=[
            transforms.InjectDefaultPrompt(None),
            *data_config.data_transforms.inputs,
            transforms.Normalize(norm_stats, use_quantiles=True),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(norm_stats, use_quantiles=True),
            *data_config.data_transforms.outputs,
        ],
        is_pytorch=True,
        pytorch_device="cuda",
    )
    adapter = Pi05MoNeContextAdapter(
        embed_dim=2048,
        num_heads=4,
        segment_size=64,
        memory_dim=args.memory_dim,
        learnable_w0=args.learnable_w0,
    )
    if args.adapter_checkpoint:
        safetensors.torch.load_model(adapter, args.adapter_checkpoint)
    elif args.memory_dim == 2048:
        with torch.no_grad():
            identity = torch.eye(2048)
            adapter.key_proj.weight.copy_(identity)
            adapter.value_proj.weight.copy_(identity)
            adapter.query_proj.weight.copy_(identity)
            adapter.output_proj.weight.copy_(identity)
    if args.injection_gate is not None:
        with torch.no_grad():
            adapter.injection_gate.fill_(args.injection_gate)
    model.attach_mone(adapter.to(device=device, dtype=torch.bfloat16))

    episode_ids = [int(value) for value in args.episodes.split(",")]
    episodes = {
        episode_id: pq.read_table(
            args.data_dir / "data" / "chunk-000" / f"episode_{episode_id:06d}.parquet"
        ).to_pylist()
        for episode_id in episode_ids
    }
    episode_meta = [
        json.loads(line)
        for line in (args.data_dir / "meta" / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    prompts = {item["episode_index"]: item["tasks"][0] for item in episode_meta}

    results = []
    fractions = np.linspace(0.2, 0.8, args.frames_per_episode)
    for episode_position, episode_id in enumerate(episode_ids):
        rows = episodes[episode_id]
        irrelevant_id = episode_ids[(episode_position + 1) % len(episode_ids)]
        irrelevant_rows = episodes[irrelevant_id]
        for frame_position, fraction in enumerate(fractions):
            current_index = min(max(10, int(len(rows) * fraction)), len(rows) - 11)
            history_index = current_index - 10
            irrelevant_index = min(history_index, len(irrelevant_rows) - 1)

            _, relevant_obs = transformed_observation(
                policy, raw_observation(rows[history_index], prompts[episode_id]), device
            )
            _, irrelevant_obs = transformed_observation(
                policy, raw_observation(irrelevant_rows[irrelevant_index], prompts[irrelevant_id]), device
            )
            current_inputs, current_obs = transformed_observation(
                policy, raw_observation(rows[current_index], prompts[episode_id]), device
            )
            relevant_state = model.build_mone_state(relevant_obs)
            irrelevant_state = model.build_mone_state(irrelevant_obs)

            generator = torch.Generator(device=device).manual_seed(2026 + episode_id * 10 + frame_position)
            noise = torch.randn(1, 10, 32, generator=generator, device=device)
            baseline = model.sample_actions(device, current_obs, noise=noise.clone(), num_steps=args.num_steps)
            relevant = model.sample_actions(
                device,
                current_obs,
                noise=noise.clone(),
                num_steps=args.num_steps,
                mone_state=relevant_state,
                memory_query_tokens=8,
            )
            irrelevant = model.sample_actions(
                device,
                current_obs,
                noise=noise.clone(),
                num_steps=args.num_steps,
                mone_state=irrelevant_state,
                memory_query_tokens=8,
            )
            baseline_actions = unnormalize_actions(policy, current_inputs, baseline)
            relevant_actions = unnormalize_actions(policy, current_inputs, relevant)
            irrelevant_actions = unnormalize_actions(policy, current_inputs, irrelevant)
            target = np.asarray(rows[current_index : current_index + 10], dtype=object)
            target = np.asarray([row["actions"] for row in target], dtype=np.float32)
            results.append(
                {
                    "episode": episode_id,
                    "task": prompts[episode_id],
                    "frame": current_index,
                    "irrelevant_episode": irrelevant_id,
                    "baseline_mse": float(np.mean((baseline_actions - target) ** 2)),
                    "relevant_mse": float(np.mean((relevant_actions - target) ** 2)),
                    "irrelevant_mse": float(np.mean((irrelevant_actions - target) ** 2)),
                }
            )
            print(f"completed {len(results)}/{len(episode_ids) * len(fractions)}", flush=True)

    baseline = np.asarray([row["baseline_mse"] for row in results])
    relevant = np.asarray([row["relevant_mse"] for row in results])
    irrelevant = np.asarray([row["irrelevant_mse"] for row in results])
    rng = np.random.default_rng(2026)
    selectivity = irrelevant - relevant
    report = {
        "checkpoint": "official pi05_libero",
        "adapter": (
            f"trained checkpoint {args.adapter_checkpoint}"
            if args.adapter_checkpoint
            else "untrained zero-W0 delta-rule, identity Q/K/V/output projections"
        ),
        "injection_gate_logit": float(adapter.injection_gate.detach()),
        "episodes": episode_ids,
        "tasks": len(episode_ids),
        "samples": len(results),
        "num_flow_steps": args.num_steps,
        "baseline_mean_mse": float(baseline.mean()),
        "conditions": [
            summary("relevant_history", relevant, baseline, rng),
            summary("irrelevant_task_history", irrelevant, baseline, rng),
        ],
        "history_selectivity": {
            "mean_irrelevant_minus_relevant_mse": float(selectivity.mean()),
            "bootstrap_95_ci": bootstrap_mean_ci(selectivity, rng),
            "relevant_better_than_irrelevant_rate_percent": float((relevant < irrelevant).mean() * 100.0),
            "wilcoxon_one_sided_p": float(stats.wilcoxon(selectivity, alternative="greater").pvalue),
        },
        "per_sample": results,
        "decision_rule": {
            "continue_if": "Relevant history improves baseline with CI above zero and outperforms irrelevant history.",
            "stop_or_redesign_if": "Relevant and irrelevant histories behave similarly, or gains are inconsistent.",
        },
    }
    pathlib.Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
