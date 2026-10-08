"""Offline action-MSE comparison on one real LIBERO demonstration episode."""

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episode", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-steps", type=int, default=2)
    args = parser.parse_args()

    device = torch.device("cuda")
    train_cfg = training_config.get_config("pi05_libero")
    model_cfg = dataclasses.replace(train_cfg.model, pytorch_compile_mode=None)
    train_cfg = dataclasses.replace(train_cfg, model=model_cfg)
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
            transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
        ],
        sample_kwargs={"num_steps": args.num_steps},
        is_pytorch=True,
        pytorch_device="cuda",
    )
    model = policy._model  # noqa: SLF001
    adapter = Pi05MoNeContextAdapter(embed_dim=2048, num_heads=4, segment_size=64)
    with torch.no_grad():
        identity = torch.eye(2048)
        adapter.key_proj.weight.copy_(identity)
        adapter.value_proj.weight.copy_(identity)
        adapter.query_proj.weight.copy_(identity)
    model.attach_mone(adapter.to(device=device, dtype=torch.bfloat16))

    rows = pq.read_table(args.episode).to_pylist()
    task_index = int(rows[0]["task_index"])
    tasks_path = pathlib.Path(args.episode).parents[2] / "meta" / "tasks.jsonl"
    tasks = [json.loads(line) for line in tasks_path.read_text(encoding="utf-8").splitlines()]
    prompt = next(item["task"] for item in tasks if item["task_index"] == task_index)

    sample_results = []
    for sample_index, current_index in enumerate((20, 80, 140)):
        history_index = current_index - 10
        history_raw = raw_observation(rows[history_index], prompt)
        current_raw = raw_observation(rows[current_index], prompt)
        _, history_obs = transformed_observation(policy, history_raw, device)
        current_inputs, current_obs = transformed_observation(policy, current_raw, device)
        state = model.build_mone_state(history_obs)

        generator = torch.Generator(device=device).manual_seed(2026 + sample_index)
        noise = torch.randn(1, 10, 32, generator=generator, device=device)
        baseline = model.sample_actions(device, current_obs, noise=noise.clone(), num_steps=args.num_steps)
        augmented = model.sample_actions(
            device,
            current_obs,
            noise=noise.clone(),
            num_steps=args.num_steps,
            mone_state=state,
            memory_query_tokens=8,
        )
        baseline_actions = unnormalize_actions(policy, current_inputs, baseline)
        mone_actions = unnormalize_actions(policy, current_inputs, augmented)
        target = np.asarray([row["actions"] for row in rows[current_index : current_index + 10]], dtype=np.float32)
        sample_results.append(
            {
                "frame": current_index,
                "history_frame": history_index,
                "baseline_mse": float(np.mean((baseline_actions - target) ** 2)),
                "mone_mse": float(np.mean((mone_actions - target) ** 2)),
                "action_mean_abs_delta": float(np.mean(np.abs(mone_actions - baseline_actions))),
            }
        )

    baseline_mean = float(np.mean([row["baseline_mse"] for row in sample_results]))
    mone_mean = float(np.mean([row["mone_mse"] for row in sample_results]))
    result = {
        "dataset": "physical-intelligence/libero",
        "episode": 0,
        "task_index": task_index,
        "prompt": prompt,
        "frames_evaluated": len(sample_results),
        "num_flow_steps": args.num_steps,
        "baseline_mean_mse": baseline_mean,
        "mone_mean_mse": mone_mean,
        "relative_mse_change_percent": (mone_mean / baseline_mean - 1.0) * 100.0,
        "samples": sample_results,
        "interpretation": "Interface/offline diagnostic only; three correlated training-episode frames are not a success-rate benchmark.",
    }
    pathlib.Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
