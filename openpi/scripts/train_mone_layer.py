"""Small LIBERO overfit experiment with frozen pi0.5 and action flow loss."""

import argparse
import json
from pathlib import Path

import jax
import numpy as np
import pyarrow.parquet as pq
import torch

from train_mone_adapter_contrastive import make_policy, raw_observation
from openpi.models import model as model_api
from openpi.models_pytorch.mone_layer import LayerMemory, capture_history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--episodes', default='0,1,2,10')
    parser.add_argument('--layer', type=int, default=9, help='Zero-based PaliGemma layer')
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--history-tokens', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--output', type=Path, default=Path('mone_layer.pt'))
    args = parser.parse_args()
    if args.steps < 1:
        parser.error('--steps must be positive')
    torch.manual_seed(2026)
    np.random.seed(2026)
    device = torch.device('cuda')
    model, policy = make_policy(args.checkpoint, device)
    layers = model.paligemma_with_expert.paligemma.language_model.layers
    if not 0 <= args.layer < len(layers):
        parser.error(f'--layer must be between 0 and {len(layers) - 1}')
    adapter = LayerMemory(layers[args.layer].self_attn.q_proj.in_features).to(device)
    metadata = [json.loads(line) for line in
                (args.data_dir / 'meta/episodes.jsonl').read_text(encoding='utf-8').splitlines()]
    prompts = {item['episode_index']: item['tasks'][0] for item in metadata}

    def observation(raw):
        inputs = policy._input_transform(raw)
        def to_tensor(value):
            array = np.asarray(value)
            # Normalization stats can promote state/actions to float64, while
            # the PyTorch pi0.5 projection layers expect float32 inputs.
            if array.dtype == np.float64:
                array = array.astype(np.float32)
            return torch.as_tensor(array, device=device)[None]

        tensors = jax.tree.map(to_tensor, inputs)
        return model_api.Observation.from_dict(tensors), tensors.get('actions')

    examples = []
    horizon = model.config.action_horizon
    for episode in map(int, args.episodes.split(',')):
        rows = pq.read_table(args.data_dir / f'data/chunk-000/episode_{episode:06d}.parquet').to_pylist()
        if len(rows) < horizon + 11:
            raise ValueError(f'Episode {episode} is too short for history and a complete action chunk')
        for index in sorted(set(np.linspace(10, len(rows) - horizon, 4).astype(int))):
            historical, _ = observation(raw_observation(rows[index - 10], prompts[episode]))
            features, mask = capture_history(model, historical, args.layer, args.history_tokens)
            raw = raw_observation(rows[index], prompts[episode])
            raw['actions'] = np.stack([row['actions'] for row in rows[index:index + horizon]])
            current, actions = observation(raw)
            if actions is None:
                raise RuntimeError('Action normalization did not preserve actions')
            examples.append((current, actions, features, mask, torch.randn_like(actions)))
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.learning_rate)
    # eval disables random augmentation/dropout; autograd is still enabled.
    # Fixed noise/time make tiny-data loss comparisons useful for debugging.
    time = torch.tensor([0.5], device=device)

    def loss_for(item, mode):
        obs, actions, history, mask, noise = item
        if mode == 'baseline':
            return model(obs, actions, noise=noise, time=time, preprocess_train=False).mean()
        if mode == 'empty':
            history = torch.zeros_like(history)
        elif mode == 'shuffled':
            history = history.roll(1, dims=1)
            mask = mask.roll(1, dims=1)
        state = adapter.build_state(history, mask)
        with adapter.activate(model, state, args.layer):
            return model(obs, actions, noise=noise, time=time, preprocess_train=False).mean()

    def evaluate():
        with torch.no_grad():
            return {mode: float(torch.stack([loss_for(item, mode) for item in examples]).mean())
                    for mode in ('baseline', 'relevant', 'empty', 'shuffled')}

    before = evaluate()
    losses = []
    for step in range(args.steps):
        obs, actions, history, mask, noise = examples[step % len(examples)]
        optimizer.zero_grad(set_to_none=True)
        state = adapter.build_state(history, mask)
        with adapter.activate(model, state, args.layer):
            loss = model(obs, actions, noise=noise, time=time, preprocess_train=False).mean()
            loss.backward()
        if not torch.isfinite(loss):
            raise RuntimeError('Non-finite action loss')
        grad = torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0, error_if_nonfinite=True)
        if grad == 0:
            raise RuntimeError('No gradient reached the memory adapter')
        optimizer.step()
        losses.append(float(loss.detach()))
        print(f'step={step + 1} action_loss={losses[-1]:.6f} grad={float(grad):.4f}', flush=True)
    report = {'layer': args.layer, 'examples': len(examples), 'before': before,
              'after': evaluate(), 'losses': losses,
              'note': 'Training-set mechanism test only; shuffled means token-order permutation, not wrong episode.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'adapter': adapter.state_dict(), 'layer': args.layer,
                'embed_dim': adapter.embed_dim, 'memory_dim': adapter.memory_dim}, args.output)
    args.output.with_suffix('.json').write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
