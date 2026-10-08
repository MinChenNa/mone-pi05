"""Pilot: train a frozen-pi0.5 L4 adapter to prefer its correct historical frame."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from eval_mone_heldout import observation, rows_for
from mone_selective_objective import selective_objective
from train_mone_adapter_contrastive import make_policy, raw_observation
from openpi.models_pytorch.mone_layer import LayerMemory, capture_history


NEGATIVE_MODES = ('same_task', 'wrong_task', 'current_frame')


def choose_frames(length, horizon, count):
    if length < horizon + 11:
        raise ValueError(f'Episode length {length} cannot supply history and action horizon')
    return sorted(set(np.linspace(10, length-horizon, count).round().astype(int).tolist()))


def relative_history_frame(current_frame, current_length, donor_length):
    return min(donor_length-1, round((current_frame-10)/(current_length-1)*(donor_length-1)))


@torch.no_grad()
def cache_examples(args, cases, model, policy, layer, device):
    rows_cache = {}

    def get_rows(episode):
        if episode not in rows_cache:
            rows_cache[episode] = rows_for(args.data_dir, episode)
        return rows_cache[episode]

    examples = []
    horizon = model.config.action_horizon
    for case in cases:
        rows = get_rows(case['episode'])
        same_rows = get_rows(case['same_episode'])
        wrong_rows = get_rows(case['wrong_episode'])
        for frame in choose_frames(len(rows), horizon, args.frames):
            raw = raw_observation(rows[frame], case['task'])
            raw['actions'] = np.asarray([r['actions'] for r in rows[frame:frame+horizon]], dtype=np.float32)
            current, actions = observation(policy._input_transform, raw, device)
            if actions is None:
                raise RuntimeError('Action transform did not preserve actions')
            sources = {
                'relevant': (rows, frame-10),
                'same_task': (same_rows, relative_history_frame(frame, len(rows), len(same_rows))),
                'wrong_task': (wrong_rows, relative_history_frame(frame, len(rows), len(wrong_rows))),
            }
            histories = {}
            for mode, (source, historical_frame) in sources.items():
                past, _ = observation(policy._input_transform,
                                      raw_observation(source[historical_frame], case['task']), device)
                histories[mode] = capture_history(model, past, layer, args.history_tokens)
            histories['current_frame'] = capture_history(model, current, layer, args.history_tokens)
            examples.append(dict(episode=case['episode'], task=case['task'], frame=frame,
                                 current=current, actions=actions, histories=histories))
        print(f"[cache] episode={case['episode']} examples={len(examples)}", flush=True)
    return examples


def action7_loss(model, adapter, example, mode, layer, noise, time):
    features, mask = example['histories'][mode]
    state = adapter.build_state(features, mask)
    with adapter.activate(model, state, layer):
        return model(example['current'], example['actions'], noise=noise,
                     time=time, preprocess_train=False)[..., :7].mean()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--source-report', type=Path, required=True)
    parser.add_argument('--initial-adapter', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--frames', type=int, default=3)
    parser.add_argument('--steps', type=int, default=360)
    parser.add_argument('--history-tokens', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--contrast-weight', type=float, default=1.0)
    parser.add_argument('--margin', type=float, default=0.002)
    parser.add_argument('--temperature', type=float, default=0.01)
    parser.add_argument('--seed', type=int, default=20260927)
    parser.add_argument('--smoke', action='store_true', help='Cache two tasks and take two steps')
    args = parser.parse_args()
    if min(args.frames, args.steps, args.history_tokens) < 1 or args.learning_rate <= 0:
        parser.error('frames, steps, history-tokens and learning-rate must be positive')
    if not torch.cuda.is_available():
        raise RuntimeError('Selective adapter training requires a GPU node')
    if args.output.exists() or args.output.with_suffix('.json').exists():
        raise FileExistsError(args.output)
    source = json.loads(args.source_report.read_text())
    if source.get('status') != 'complete' or len(source['cases']) < 4:
        raise ValueError('Source report must be a completed multi-task evaluation')
    cases = source['cases'][:4] if args.smoke else source['cases']
    if len({case['task'] for case in cases}) < 2:
        raise ValueError('Need at least two source tasks')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda:0')
    model, policy = make_policy(args.checkpoint, device)
    initial = torch.load(args.initial_adapter, map_location='cpu', weights_only=True)
    layer = int(initial['layer'])
    adapter = LayerMemory(initial['embed_dim'], initial['memory_dim']).to(device)
    adapter.load_state_dict(initial['adapter'], strict=True)
    examples = cache_examples(args, cases, model, policy, layer, device)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.learning_rate)
    rng = np.random.default_rng(args.seed)
    order = []
    steps = 2 if args.smoke else args.steps
    while len(order) < steps:
        order.extend(rng.permutation(len(examples)).tolist())
    metrics = []
    for step, index in enumerate(order[:steps]):
        example = examples[index]
        epoch = step // len(examples)
        negative_mode = NEGATIVE_MODES[(epoch + index) % len(NEGATIVE_MODES)]
        draw = epoch % 3
        generator = torch.Generator(device=device).manual_seed(
            args.seed + example['episode']*10000 + example['frame']*10 + epoch)
        noise = torch.randn(example['actions'].shape, generator=generator, device=device,
                            dtype=example['actions'].dtype)
        time = torch.tensor([(draw+0.5)/3], device=device, dtype=torch.float32)
        with torch.no_grad():
            baseline = model(example['current'], example['actions'], noise=noise,
                             time=time, preprocess_train=False)[..., :7].mean()
        optimizer.zero_grad(set_to_none=True)
        positive = action7_loss(model, adapter, example, 'relevant', layer, noise, time)
        negative = action7_loss(model, adapter, example, negative_mode, layer, noise, time)
        total, ranking = selective_objective(
            positive, negative, baseline, margin=args.margin,
            temperature=args.temperature, contrast_weight=args.contrast_weight)
        if not all(torch.isfinite(x) for x in (positive, negative, total)):
            raise RuntimeError(f'Nonfinite loss at step {step+1}')
        total.backward()
        grad = torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0, error_if_nonfinite=True)
        if grad == 0:
            raise RuntimeError('No gradient reached the adapter')
        optimizer.step()
        row = dict(step=step+1, episode=example['episode'], negative_mode=negative_mode,
                   positive=float(positive.detach()), negative=float(negative.detach()),
                   baseline=float(baseline), time=float(time), ranking=float(ranking.detach()),
                   total=float(total.detach()), grad=float(grad))
        metrics.append(row)
        if args.smoke or (step+1) % 20 == 0 or step == 0:
            print(f"[train] step={step+1}/{steps} negative={negative_mode} "
                  f"positive={row['positive']:.6f} negative_loss={row['negative']:.6f} "
                  f"baseline={row['baseline']:.6f} rank={row['ranking']:.6f} "
                  f"grad={row['grad']:.4f}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(adapter=adapter.state_dict(), layer=layer, embed_dim=adapter.embed_dim,
                    memory_dim=adapter.memory_dim, training_method='bounded_action7_selective',
                    initial_adapter_sha256=hashlib.sha256(args.initial_adapter.read_bytes()).hexdigest()),
               args.output)
    report = dict(status='complete', smoke=args.smoke, layer=layer, steps=steps,
                  examples=len(examples), tasks=sorted({case['task'] for case in cases}),
                  episodes=sorted({case['episode'] for case in cases}),
                  source_report=str(args.source_report), initial_adapter=str(args.initial_adapter),
                  output=str(args.output), negative_modes=NEGATIVE_MODES,
                  objective=dict(margin=args.margin, temperature=args.temperature,
                                 contrast_weight=args.contrast_weight,
                                 negative_loss_cap='frozen backbone baseline'),
                  metrics=metrics,
                  note='Training metrics are not evidence of selectivity; evaluate on excluded tasks and compare controls.')
    args.output.with_suffix('.json').write_text(json.dumps(report, indent=2))
    print(f'[done] adapter={args.output} report={args.output.with_suffix(".json")}', flush=True)


if __name__ == '__main__':
    main()
