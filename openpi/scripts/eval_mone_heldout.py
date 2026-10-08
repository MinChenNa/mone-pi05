"""Paired held-out adapter diagnostics; never performs optimization."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import jax
import numpy as np
import pyarrow.parquet as pq
import torch

if __package__:
    from .train_mone_adapter_contrastive import make_policy, raw_observation
else:
    from train_mone_adapter_contrastive import make_policy, raw_observation
from openpi import transforms
from openpi.models import model as model_api
from openpi.models_pytorch.mone_layer import LayerMemory, capture_history
from openpi.policies import libero_policy
from openpi.shared import normalize
from openpi.training import config


TRAIN_EPISODES = {0, 1, 2, 10}
MODES = ('baseline', 'relevant', 'wrong_task', 'same_task', 'current_frame',
         'fixed_history', 'empty', 'roll', 'reverse_valid')


def reverse_valid(history, mask):
    result = history.clone()
    for batch in range(len(history)):
        indices = mask[batch].nonzero().flatten()
        result[batch, indices] = history[batch, indices.flip(0)]
    return result


def make_transform(checkpoint):
    cfg = config.get_config('pi05_libero').model
    return transforms.compose([
        transforms.InjectDefaultPrompt(None),
        libero_policy.LiberoInputs(model_type=cfg.model_type),
        transforms.Normalize(normalize.load(checkpoint / 'assets/physical-intelligence/libero'), use_quantiles=True),
        *config.ModelTransformFactory()(cfg).inputs,
    ])


def observation(transform, raw, device):
    def tensor(value):
        array = np.asarray(value)
        if array.dtype == np.float64:
            array = array.astype(np.float32)
        return torch.as_tensor(array, device=device)[None]
    data = jax.tree.map(tensor, transform(raw))
    return model_api.Observation.from_dict(data), data.get('actions')


def rows_for(root, episode):
    info = json.loads((root / 'meta/info.json').read_text())
    chunk = episode // info.get('chunks_size', 1000)
    pattern = info.get('data_path', 'data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet')
    path = root / pattern.format(episode_chunk=chunk, episode_index=episode)
    return pq.read_table(path).to_pylist()


def plan(root, task_count, seed, excluded_tasks=()):
    metadata = [json.loads(x) for x in (root / 'meta/episodes.jsonl').read_text().splitlines()]
    groups = defaultdict(list)
    for item in metadata:
        if item['episode_index'] not in TRAIN_EPISODES and item['length'] >= 30:
            groups[item['tasks'][0]].append(item['episode_index'])
    tasks = sorted(k for k, v in groups.items() if len(v) >= 3 and k not in excluded_tasks)
    if len(tasks) < task_count or task_count < 2:
        raise ValueError('Need >=2 tasks and >=3 held-out episodes per task')
    selected = np.linspace(0, len(tasks)-1, task_count).round().astype(int)
    rng = np.random.default_rng(seed)
    chosen = [(tasks[i], rng.permutation(sorted(groups[tasks[i]]))[:3].tolist()) for i in selected]
    result = []
    for i, (task, episodes) in enumerate(chosen):
        other_task, other_episodes = chosen[(i+1) % len(chosen)]
        for episode in episodes[:2]:
            result.append(dict(episode=episode, task=task, same_episode=episodes[2],
                               wrong_episode=other_episodes[2], wrong_task=other_task))
    return result


def summarize(records, loss_key='losses', group_key='episode'):
    # Average noise draws and frames within each episode or task before bootstrapping.
    grouped = defaultdict(list)
    for row in records:
        grouped[row[group_key]].append(row)
    means = {mode: np.array([np.mean([r[loss_key][mode] for r in rows])
                            for rows in grouped.values()]) for mode in MODES}
    rng = np.random.default_rng(743)
    indices = rng.integers(len(grouped), size=(5000, len(grouped)))
    result = {f'{group_key}_count': len(grouped),
              'mean_loss': {k: float(v.mean()) for k, v in means.items()}}
    for name, control in [('gain_vs_baseline', 'baseline'), ('gain_vs_wrong_task', 'wrong_task'),
                          ('gain_vs_same_task', 'same_task'),
                          ('gain_vs_current_frame', 'current_frame'),
                          ('gain_vs_fixed_history', 'fixed_history')]:
        delta = means[control] - means['relevant']
        ci = np.quantile(delta[indices].mean(axis=1), [.025, .975])
        result[name] = dict(mean=float(delta.mean()), ci95=ci.tolist(),
                            **{f'{group_key}_win_fraction': float((delta > 0).mean())})
    result[f'per_{group_key}'] = {str(key): {mode: float(means[mode][i]) for mode in MODES}
                                  for i, key in enumerate(grouped)}
    result['ci_note'] = f'Paired {group_key} bootstrap, descriptive; one adapter-training seed.'
    return result


def audit_cpu(args, cases, transform):
    # Reconstruct the exact prefix validity mask without loading the backbone.
    from transformers import PaliGemmaConfig
    vision = PaliGemmaConfig().vision_config
    image_tokens = (vision.image_size // vision.patch_size) ** 2
    masks = []
    metadata = [json.loads(x) for x in (args.data_dir / 'meta/episodes.jsonl').read_text().splitlines()]
    audit_cases = cases + [dict(episode=x['episode_index'], task=x['tasks'][0])
                          for x in metadata if x['episode_index'] in TRAIN_EPISODES]
    for case in audit_cases:
        raw = {'observation/state': np.zeros(8, np.float32),
               'observation/image': np.zeros((256,256,3), np.uint8),
               'observation/wrist_image': np.zeros((256,256,3), np.uint8), 'prompt': case['task']}
        inputs = transform(raw)
        valid = torch.cat([torch.full((image_tokens,), bool(v)) for v in inputs['image_mask'].values()]
                          + [torch.as_tensor(inputs['tokenized_prompt_mask'])])
        indices = torch.linspace(0, len(valid)-1, min(args.history_tokens, len(valid))).round().long()
        mask = valid[indices]
        order = torch.arange(len(mask))
        unchanged = torch.equal(order[mask], order.roll(1)[mask.roll(1)])
        masks.append(dict(episode=case['episode'], adapter_training_episode=case['episode'] in TRAIN_EPISODES,
                          last_token_valid=bool(mask[-1]),
                          valid_tokens=int(mask.sum()), roll_preserves_valid_order=unchanged))
    checkpoint = torch.load(args.adapter, map_location='cpu', weights_only=True)
    adapter = LayerMemory(checkpoint['embed_dim'], checkpoint['memory_dim']).eval()
    adapter.load_state_dict(checkpoint['adapter'], strict=True)
    torch.manual_seed(args.seed)
    history = torch.randn(1, args.history_tokens, checkpoint['embed_dim'])
    mask = torch.ones(1, args.history_tokens, dtype=torch.bool)
    mask[:, -1] = False
    with torch.no_grad():
        state = adapter.build_state(history, mask).weights
        rolled = adapter.build_state(history.roll(1,1), mask.roll(1,1)).weights
        reversed_state = adapter.build_state(reverse_valid(history, mask), mask).weights
    return dict(actual_prompt_masks=masks, synthetic_features_trained_adapter=True,
                padded_roll_max_state_difference=float((state-rolled).abs().max()),
                valid_reverse_max_state_difference=float((state-reversed_state).abs().max()),
                note='Mask check uses real task prompts and the same transforms. State comparison uses synthetic features, not real backbone outputs.')


@torch.no_grad()
def evaluate(args, cases):
    if not torch.cuda.is_available():
        raise RuntimeError('Held-out backbone evaluation requires a GPU node; use --prepare-only on login node')
    device = torch.device('cuda:0')
    model, policy = make_policy(args.checkpoint, device)
    checkpoint = torch.load(args.adapter, map_location='cpu', weights_only=True)
    layer = checkpoint['layer']
    adapter = LayerMemory(checkpoint['embed_dim'], checkpoint['memory_dim']).to(device).eval()
    adapter.load_state_dict(checkpoint['adapter'], strict=True)
    records = []
    horizon = model.config.action_horizon
    fixed_episode = cases[0]['same_episode']
    fixed_task = cases[0]['task']
    fixed_frame = 10
    fixed_rows = rows_for(args.data_dir, fixed_episode)
    fixed_obs, _ = observation(policy._input_transform,
                               raw_observation(fixed_rows[fixed_frame], fixed_task), device)
    fixed_features, fixed_mask = capture_history(model, fixed_obs, layer, args.history_tokens)
    fixed_state = adapter.build_state(fixed_features, fixed_mask)
    with args.output.with_suffix('.jsonl').open('x') as output:
        for case in cases:
            rows = rows_for(args.data_dir, case['episode'])
            same_rows = rows_for(args.data_dir, case['same_episode'])
            wrong_rows = rows_for(args.data_dir, case['wrong_episode'])
            frames = np.linspace(10, len(rows)-horizon, args.frames).round().astype(int)
            for frame in sorted(set(frames.tolist())):
                raw = raw_observation(rows[frame], case['task'])
                raw['actions'] = np.asarray([r['actions'] for r in rows[frame:frame+horizon]], dtype=np.float32)
                current, actions = observation(policy._input_transform, raw, device)
                states = {}
                histories = {}
                history_episodes = {}
                sources = [('relevant', rows, case['episode'], frame-10),
                           ('same_task', same_rows, case['same_episode'],
                            min(len(same_rows)-1, round((frame-10)/(len(rows)-1)*(len(same_rows)-1)))),
                           ('wrong_task', wrong_rows, case['wrong_episode'],
                            min(len(wrong_rows)-1, round((frame-10)/(len(rows)-1)*(len(wrong_rows)-1)))),
                           ('current_frame', rows, case['episode'], frame)]
                for mode, source, source_episode, history_frame in sources:
                    # Swap visual/state content while keeping the current task prompt fixed.
                    if mode == 'current_frame':
                        past = current
                    else:
                        past, _ = observation(policy._input_transform,
                                              raw_observation(source[history_frame], case['task']), device)
                    features, mask = capture_history(model, past, layer, args.history_tokens)
                    histories[mode] = history_frame
                    history_episodes[mode] = source_episode
                    states[mode] = adapter.build_state(features, mask)
                    if mode == 'relevant':
                        states['roll'] = adapter.build_state(features.roll(1,1), mask.roll(1,1))
                        states['reverse_valid'] = adapter.build_state(reverse_valid(features, mask), mask)
                        states['empty'] = adapter.empty_state(1, device=device, dtype=features.dtype)
                states['fixed_history'] = fixed_state
                histories['fixed_history'] = fixed_frame
                history_episodes['fixed_history'] = fixed_episode
                diagnostics = {mode: float((states[mode].weights-states['relevant'].weights).abs().max())
                               for mode in ('roll','reverse_valid','same_task','wrong_task',
                                            'current_frame','fixed_history')}
                for draw in range(args.draws):
                    seed = args.seed + case['episode']*10000 + frame*10 + draw
                    gen = torch.Generator(device=device).manual_seed(seed)
                    noise = torch.randn(actions.shape, generator=gen, device=device, dtype=actions.dtype)
                    time = torch.tensor([(draw+0.5)/args.draws], device=device, dtype=torch.float32)
                    losses = {}
                    losses_action7 = {}
                    for mode in MODES:
                        if mode == 'baseline':
                            loss_tensor = model(current, actions, noise=noise, time=time, preprocess_train=False)
                        else:
                            with adapter.activate(model, states[mode], layer):
                                loss_tensor = model(current, actions, noise=noise, time=time, preprocess_train=False)
                        loss = loss_tensor.mean()
                        if not torch.isfinite(loss):
                            raise RuntimeError(f'Nonfinite loss: {case}, {frame}, {mode}')
                        losses[mode] = float(loss)
                        losses_action7[mode] = float(loss_tensor[..., :7].mean())
                    if abs(losses['empty']-losses['baseline']) > 1e-5:
                        raise RuntimeError('Empty memory changed baseline; inspect hook identity before interpreting results')
                    record = dict(**case, frame=frame, history_frames=histories,
                                  history_episodes=history_episodes, seed=seed,
                                  time=float(time), losses=losses, losses_action7=losses_action7,
                                  state_max_differences=diagnostics)
                    records.append(record)
                    output.write(json.dumps(record)+'\n')
                    output.flush()
                print(f"[eval] layer={layer} episode={case['episode']} frame={frame} records={len(records)}", flush=True)
    return dict(layer=layer, fixed_history_source=dict(episode=fixed_episode, task=fixed_task,
                                                       frame=fixed_frame),
                summary=summarize(records), summary_action7=summarize(records, 'losses_action7'),
                summary_task=summarize(records, group_key='task'),
                summary_action7_task=summarize(records, 'losses_action7', 'task'),
                note='Held out from adapter training only; base-model exposure unknown. Flow-matching loss, not rollout success.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--adapter', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tasks', type=int, default=8)
    parser.add_argument('--frames', type=int, default=3)
    parser.add_argument('--draws', type=int, default=3)
    parser.add_argument('--history-tokens', type=int, default=64)
    parser.add_argument('--seed', type=int, default=20260926)
    parser.add_argument('--exclude-report', type=Path, action='append', default=[],
                        help='Completed report whose tasks must be excluded; may be repeated')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    if min(args.frames, args.draws, args.history_tokens) < 1:
        parser.error('frames, draws and history-tokens must be positive')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    excluded_tasks = set()
    for report_path in args.exclude_report:
        prior = json.loads(report_path.read_text())
        if prior.get('status') != 'complete':
            raise ValueError(f'Exclusion report is not complete: {report_path}')
        excluded_tasks.update(case['task'] for case in prior['cases'])
    cases = plan(args.data_dir, args.tasks, args.seed, excluded_tasks)
    result = dict(adapter=str(args.adapter.resolve()),
                  adapter_sha256=hashlib.sha256(args.adapter.read_bytes()).hexdigest(),
                  excluded_adapter_training_episodes=sorted(TRAIN_EPISODES),
                  excluded_prior_evaluation_tasks=sorted(excluded_tasks), cases=cases,
                  settings={k:([str(p) for p in v] if isinstance(v, list) else
                               str(v) if isinstance(v, Path) else v)
                            for k,v in vars(args).items()})
    if args.prepare_only:
        # Verify parquet availability and required columns without loading the model.
        for episode in sorted({c[k] for c in cases for k in ('episode','same_episode','wrong_episode')}):
            rows = rows_for(args.data_dir, episode)
            if not {'state','actions','image','wrist_image'} <= rows[0].keys():
                raise ValueError(f'Missing columns in episode {episode}')
        result['cpu_audit'] = audit_cpu(args, cases, make_transform(args.checkpoint))
        result['status'] = 'prepared_not_gpu_evaluated'
    else:
        result.update(evaluate(args, cases))
        result['status'] = 'complete'
    args.output.write_text(json.dumps(result, indent=2))
    print(f'[done] {args.output}', flush=True)


if __name__ == '__main__':
    main()
