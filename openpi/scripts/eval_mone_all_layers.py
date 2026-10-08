"""Paired held-out action loss for 18-layer Mone and matched-size control."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

if __package__:
    from .eval_mone_heldout import observation, rows_for
    from .plan_mone_full import digest
    from .train_mone_adapter_contrastive import make_policy, raw_observation
else:
    from eval_mone_heldout import observation, rows_for
    from plan_mone_full import digest
    from train_mone_adapter_contrastive import make_policy, raw_observation
from openpi.models_pytorch.mone_all_layers import AllLayerMemory, capture_context_layers


MODES = ('baseline', 'relevant', 'same_task', 'fixed_history',
         'current_frame', 'parameter_control')


def file_sha256(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def load_adapter(path, plan_hash, backbone_hash, device):
    saved = torch.load(path, map_location='cpu', weights_only=False)
    if saved['plan_sha256'] != plan_hash or saved['backbone_sha256'] != backbone_hash:
        raise ValueError(f'Adapter provenance mismatch: {path}')
    if saved['history_tokens'] < 1 or saved['memory_dim'] < 1:
        raise ValueError('Invalid adapter dimensions')
    adapter = AllLayerMemory(memory_dim=saved['memory_dim']).to(device).eval()
    adapter.load_state_dict(saved['adapter'], strict=True)
    return adapter, saved


@torch.no_grad()
def evaluate(args, plan, memory, control, model, policy, device, stream):
    tasks = sorted(plan['partitions'][args.split])[args.shard::args.shards]
    fixed_task = sorted(plan['partitions']['train'])[0]
    fixed_episode = plan['episodes'][fixed_task][0]['episode']
    fixed_rows = rows_for(args.data_dir, fixed_episode)
    fixed_obs, _ = observation(policy._input_transform,
                               raw_observation(fixed_rows[10], fixed_task), device)
    fixed_features, fixed_mask = capture_context_layers(model, fixed_obs, args.history_tokens)
    count = 0
    for task in tasks:
        episodes = plan['episodes'][task]
        donor_rows = rows_for(args.data_dir, episodes[2]['episode'])
        for source in episodes[:2]:
            rows = rows_for(args.data_dir, source['episode'])
            frames = np.linspace(10, len(rows)-model.config.action_horizon,
                                 args.frames).round().astype(int)
            for frame in sorted(set(frames.tolist())):
                raw = raw_observation(rows[frame], task)
                raw['actions'] = np.asarray([r['actions'] for r in
                                             rows[frame:frame+model.config.action_horizon]],
                                            dtype=np.float32)
                current, actions = observation(policy._input_transform, raw, device)
                relevant_obs, _ = observation(policy._input_transform,
                                              raw_observation(rows[frame-10], task), device)
                donor_frame = min(len(donor_rows)-1,
                                  round((frame-10)/(len(rows)-1)*(len(donor_rows)-1)))
                wrong_obs, _ = observation(policy._input_transform,
                                           raw_observation(donor_rows[donor_frame], task), device)
                relevant, relevant_mask = capture_context_layers(model, relevant_obs,
                                                                  args.history_tokens)
                wrong, wrong_mask = capture_context_layers(model, wrong_obs, args.history_tokens)
                own, own_mask = capture_context_layers(model, current, args.history_tokens)
                if not (torch.equal(relevant_mask, wrong_mask) and
                        torch.equal(relevant_mask, own_mask)):
                    raise RuntimeError('Historical prompt masks differ in paired evaluation')
                for draw in range(args.draws):
                    seed = args.seed + source['episode']*10000 + frame*10 + draw
                    generator = torch.Generator(device=device).manual_seed(seed)
                    noise = torch.randn(actions.shape, generator=generator,
                                        device=device, dtype=actions.dtype)
                    time = torch.tensor([(draw+.5)/args.draws], device=device, dtype=torch.float32)
                    losses = {'baseline': float(model(current, actions, noise=noise, time=time,
                                                       preprocess_train=False)[..., :7].mean())}
                    for mode, features, mask, adapter in (
                        ('relevant', relevant, relevant_mask, memory),
                        ('same_task', wrong, wrong_mask, memory),
                        ('fixed_history', fixed_features, fixed_mask, memory),
                        ('current_frame', own, own_mask, memory),
                        ('parameter_control', own, own_mask, control),
                    ):
                        losses[mode] = float(adapter(model, current, actions, features, mask,
                                                     noise, time))
                    if not all(np.isfinite(value) for value in losses.values()):
                        raise RuntimeError(f'Nonfinite paired losses: {task} episode={source["episode"]}')
                    row = dict(split=args.split, task=task, episode=source['episode'],
                               frame=int(frame), draw=draw, seed=seed,
                               same_task_donor=episodes[2]['episode'],
                               fixed_donor=fixed_episode, losses=losses)
                    stream.write(json.dumps(row)+'\n')
                    stream.flush()
                    count += 1
            print(f"[eval] task={task} episode={source['episode']} records={count}", flush=True)
    return tasks, count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--memory-adapter', type=Path, required=True)
    parser.add_argument('--control-adapter', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--split', choices=('val', 'test'), default='test')
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=4)
    parser.add_argument('--frames', type=int, default=3)
    parser.add_argument('--draws', type=int, default=2)
    parser.add_argument('--history-tokens', type=int, default=32)
    parser.add_argument('--seed', type=int, default=20260927)
    args = parser.parse_args()
    if min(args.shards, args.frames, args.draws, args.history_tokens) < 1 or not 0 <= args.shard < args.shards:
        parser.error('Invalid counts')
    if args.output.exists() or args.output.with_suffix('.jsonl').exists():
        raise FileExistsError(args.output)
    if not torch.cuda.is_available():
        raise RuntimeError('GPU evaluation required')
    plan = json.loads(args.plan.read_text())
    plan_hash = plan.pop('plan_sha256')
    if digest(plan) != plan_hash:
        raise ValueError('Plan hash mismatch')
    device = torch.device('cuda:0')
    model, policy = make_policy(args.checkpoint, device)
    memory_raw = torch.load(args.memory_adapter, map_location='cpu', weights_only=False)
    backbone_hash = memory_raw['backbone_sha256']
    if file_sha256(args.checkpoint/'model.safetensors') != backbone_hash:
        raise ValueError('Evaluation backbone differs from trained checkpoint')
    memory, memory_meta = load_adapter(args.memory_adapter, plan_hash, backbone_hash, device)
    control, control_meta = load_adapter(args.control_adapter, plan_hash, backbone_hash, device)
    if (memory_meta['history_mode'], control_meta['history_mode']) != (
        'true_history', 'current_frame'):
        raise ValueError('Expected trained true-history and current-frame adapters')
    if memory_meta['history_tokens'] != args.history_tokens or control_meta['history_tokens'] != args.history_tokens:
        raise ValueError('History token count differs from training')
    if memory_meta['step'] != control_meta['step'] or memory_meta['seed'] != control_meta['seed']:
        raise ValueError('Parameter control did not receive matched steps and seed')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.with_suffix('.jsonl').open('x') as stream:
        tasks, count = evaluate(args, plan, memory, control, model, policy, device, stream)
    args.output.write_text(json.dumps(dict(status='complete', split=args.split,
                                           shard=args.shard, shards=args.shards,
                                           checkpoint=str(args.checkpoint.resolve()),
                                           plan_sha256=plan_hash, backbone_sha256=backbone_hash,
                                           memory_adapter=str(args.memory_adapter),
                                           control_adapter=str(args.control_adapter),
                                           tasks=tasks, records=count, frames=args.frames,
                                           draws=args.draws), indent=2))
    print(f'[done] {args.output} records={count}', flush=True)


if __name__ == '__main__':
    main()
