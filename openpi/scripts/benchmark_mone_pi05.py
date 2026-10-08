"""Small CPU benchmark for the MoNe x pi0.5 mechanism prototype."""

import argparse
import json
import time

import torch
import torch.nn.functional as F  # noqa: N812

from openpi.models_pytorch.mone_pi05 import DeltaFastWeightMemory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="mone_pi05_benchmark.json")
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--query-tokens", type=int, default=8)
    args = parser.parse_args()

    torch.manual_seed(2026)
    memory = DeltaFastWeightMemory(args.embed_dim, num_heads=4)

    # Direct associative-effect check on a lightly loaded memory.
    keys = torch.randn(1, 8, args.embed_dim)
    values = torch.randn_like(keys)
    empty = memory.empty_state(1, device=keys.device, dtype=keys.dtype)
    written = memory.write_segment(empty, keys, values)
    recalled = memory.read(written, keys)
    target = F.silu(values)
    recall_cosine = F.cosine_similarity(recalled, target, dim=-1).mean().item()

    scaling = []
    query = torch.randn(1, args.query_tokens, args.embed_dim)
    for history_tokens in (128, 512, 2048):
        history = torch.randn(1, history_tokens, args.embed_dim)
        state = memory.empty_state(1, device=history.device, dtype=history.dtype)
        write_start = time.perf_counter()
        state = memory.write_segment(state, history, history)
        write_ms = (time.perf_counter() - write_start) * 1000

        for _ in range(10):
            memory.read(state, query)
        read_start = time.perf_counter()
        for _ in range(100):
            memory.read(state, query)
        read_ms = (time.perf_counter() - read_start) * 10
        scaling.append(
            {
                "history_tokens": history_tokens,
                "state_bytes": state.weights.numel() * state.weights.element_size(),
                "memory_query_tokens": args.query_tokens,
                "write_ms": round(write_ms, 3),
                "mean_read_ms": round(read_ms, 3),
            }
        )

    result = {
        "seed": 2026,
        "device": "cpu",
        "recall_cosine_after_write": round(recall_cosine, 6),
        "recall_cosine_empty_baseline": 0.0,
        "scaling": scaling,
    }
    with open(args.output, "w", encoding="utf-8") as output_file:
        json.dump(result, output_file, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
