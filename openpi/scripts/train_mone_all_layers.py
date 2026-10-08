"""Four-GPU DDP training of independent memory blocks at all pi0.5 context layers."""

import argparse
from collections import OrderedDict
from contextlib import nullcontext
import hashlib
import json
import os
import time as wall_time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

if __package__:
    from .eval_mone_heldout import observation, rows_for
    from .plan_mone_full import digest
    from .train_mone_adapter_contrastive import make_policy, raw_observation
else:
    from eval_mone_heldout import observation, rows_for
    from plan_mone_full import digest
    from train_mone_adapter_contrastive import make_policy, raw_observation
from openpi.models_pytorch.mone_all_layers import AllLayerMemory, capture_context_layers


class EpisodeSampler:
    def __init__(self, data_dir, plan, split, cache_size=3):
        self.data_dir = data_dir
        self.tasks = plan['partitions'][split]
        self.episodes = plan['episodes']
        self.cache_size = cache_size
        self.cache = OrderedDict()

    def rows(self, episode):
        if episode not in self.cache:
            self.cache[episode] = rows_for(self.data_dir, episode)
            if len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
        self.cache.move_to_end(episode)
        return self.cache[episode]

    def sample(self, rng, horizon, history_mode):
        task = self.tasks[int(rng.integers(len(self.tasks)))]
        sources = self.episodes[task]
        current_index = int(rng.integers(len(sources)))
        donor_index = int(rng.integers(len(sources)-1))
        if donor_index >= current_index:
            donor_index += 1
        episode = sources[current_index]['episode']
        donor = sources[donor_index]['episode']
        rows = self.rows(episode)
        donor_rows = self.rows(donor)
        frame = int(rng.integers(10, len(rows)-horizon+1))
        donor_frame = min(len(donor_rows)-1, round((frame-10)/(len(rows)-1)*(len(donor_rows)-1)))
        current = raw_observation(rows[frame], task)
        current['actions'] = np.asarray([r['actions'] for r in rows[frame:frame+horizon]],
                                        dtype=np.float32)
        positive_frame = frame-10 if history_mode == 'true_history' else frame
        positive = raw_observation(rows[positive_frame], task)
        negative = raw_observation(donor_rows[donor_frame], task)
        return dict(task=task, episode=episode, donor=donor, frame=frame,
                    current=current, positive=positive, negative=negative)


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def save_checkpoint(path, adapter, optimizer, step, args, plan_hash, backbone_hash):
    payload = dict(step=step, adapter=adapter.state_dict(), optimizer=optimizer.state_dict(),
                   plan_sha256=plan_hash, backbone_sha256=backbone_hash,
                   history_mode=args.history_mode,
                   history_tokens=args.history_tokens, memory_dim=args.memory_dim,
                   seed=args.seed, gradient_accumulation=args.accumulation,
                   world_size=dist.get_world_size())
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(payload, temporary)
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--history-mode', choices=('true_history', 'current_frame'), required=True)
    parser.add_argument('--steps', type=int, default=3000)
    parser.add_argument('--accumulation', type=int, default=8)
    parser.add_argument('--history-tokens', type=int, default=32)
    parser.add_argument('--memory-dim', type=int, default=256)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--save-every', type=int, default=250)
    parser.add_argument('--seed', type=int, default=20260927)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if min(args.steps, args.accumulation, args.history_tokens, args.memory_dim,
           args.save_every) < 1 or args.learning_rate <= 0:
        parser.error('Invalid training settings')
    rank = int(os.environ.get('RANK', '0'))
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    if world_size != 4 or not torch.cuda.is_available():
        raise RuntimeError('This training plan requires torchrun on four CUDA GPUs')
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    dist.init_process_group('nccl', device_id=device)
    try:
        plan = json.loads(args.plan.read_text())
        plan_hash = plan.pop('plan_sha256')
        if digest(plan) != plan_hash:
            raise ValueError('Data plan changed after preparation')
        torch.manual_seed(args.seed)
        model, policy = make_policy(args.checkpoint, device)
        if model.training or any(p.requires_grad for p in model.parameters()):
            raise RuntimeError('Backbone must stay frozen and in eval mode')
        depth = len(model.paligemma_with_expert.paligemma.language_model.layers)
        if depth != 18:
            raise ValueError(f'Expected pi0.5 context depth 18, got {depth}')
        if rank == 0:
            backbone_hash = file_sha256(args.checkpoint/'model.safetensors')
        else:
            backbone_hash = None
        values = [backbone_hash]
        dist.broadcast_object_list(values, src=0)
        backbone_hash = values[0]
        adapter = AllLayerMemory(depth=depth, memory_dim=args.memory_dim).to(device)
        optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.learning_rate)
        start_step = 0
        if args.resume:
            saved = torch.load(args.resume, map_location='cpu', weights_only=False)
            if (saved['plan_sha256'], saved['backbone_sha256'], saved['history_mode'],
                saved['history_tokens'], saved['memory_dim'],
                    saved['world_size']) != (plan_hash, backbone_hash, args.history_mode,
                args.history_tokens, args.memory_dim, world_size):
                raise ValueError('Resume checkpoint does not match this experiment')
            if (saved['seed'], saved['gradient_accumulation']) != (args.seed, args.accumulation):
                raise ValueError('Resume seed and gradient accumulation must match the checkpoint')
            adapter.load_state_dict(saved['adapter'], strict=True)
            optimizer.load_state_dict(saved['optimizer'])
            start_step = int(saved['step'])
        ddp = DDP(adapter, device_ids=[local_rank], output_device=local_rank,
                  broadcast_buffers=False)
        sampler = EpisodeSampler(args.data_dir, plan, 'train')
        args.run_dir.mkdir(parents=True, exist_ok=True)
        # Keep interrupted attempts separate: the last metrics may be ahead of
        # the saved optimizer state. Reusing them would duplicate step numbers.
        metrics_path = args.run_dir / (
            f'rank{rank}.jsonl' if not args.resume else
            f'rank{rank}.resume{start_step:06d}.{wall_time.time_ns()}.jsonl')
        if metrics_path.exists():
            raise FileExistsError(metrics_path)
        steps = min(args.steps, start_step+2) if args.smoke else args.steps
        with metrics_path.open('x') as metrics:
            for step in range(start_step, steps):
                optimizer.zero_grad(set_to_none=True)
                loss_sum = pos_sum = neg_sum = 0.0
                for micro in range(args.accumulation):
                    rng = np.random.default_rng(args.seed + step*100000 + rank*1000 + micro)
                    item = sampler.sample(rng, model.config.action_horizon, args.history_mode)
                    current, actions = observation(policy._input_transform, item['current'], device)
                    positive_obs, _ = observation(policy._input_transform, item['positive'], device)
                    negative_obs, _ = observation(policy._input_transform, item['negative'], device)
                    pos_features, mask = capture_context_layers(model, positive_obs, args.history_tokens)
                    neg_features, neg_mask = capture_context_layers(model, negative_obs, args.history_tokens)
                    if not torch.equal(mask, neg_mask):
                        raise RuntimeError('Historical prompt masks differ for paired samples')
                    generator = torch.Generator(device=device).manual_seed(
                        args.seed + step*100000 + rank*1000 + micro)
                    noise = torch.randn(actions.shape, generator=generator, device=device,
                                        dtype=actions.dtype)
                    time = torch.tensor([float(rng.uniform(.02, .98))],
                                        device=device, dtype=torch.float32)
                    with torch.no_grad():
                        baseline = model(current, actions, noise=noise, time=time,
                                         preprocess_train=False)[..., :7].mean()
                        if step == start_step and micro == 0:
                            empty = adapter(model, current, actions, pos_features, mask,
                                            noise, time, empty=True)
                            delta = float((empty - baseline).abs())
                            if delta > 1e-5:
                                raise RuntimeError(
                                    f'Empty Mone memory changes frozen backbone loss: {delta:.8f}')
                            print(f'[preflight] rank={rank} empty-memory identity delta={delta:.8f}',
                                  flush=True)
                    sync = nullcontext() if micro == args.accumulation-1 else ddp.no_sync()
                    with sync:
                        total, pos, neg = ddp(model, current, actions, pos_features, mask,
                                              noise, time, negative_features=neg_features,
                                              baseline=baseline)
                        if not torch.isfinite(total):
                            raise RuntimeError(f'Nonfinite loss at step {step+1}')
                        (total / args.accumulation).backward()
                    loss_sum += float(total.detach())
                    pos_sum += float(pos)
                    neg_sum += float(neg)
                grad = torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0,
                                                      error_if_nonfinite=True)
                optimizer.step()
                row = dict(step=step+1, rank=rank, total=loss_sum/args.accumulation,
                           relevant=pos_sum/args.accumulation,
                           same_task_negative=neg_sum/args.accumulation, grad=float(grad))
                metrics.write(json.dumps(row) + '\n')
                metrics.flush()
                if rank == 0 and ((step+1) % 20 == 0 or step == start_step):
                    print(f"[train] step={step+1}/{steps} loss={row['total']:.6f} "
                          f"positive={row['relevant']:.6f} negative={row['same_task_negative']:.6f} "
                          f"grad={row['grad']:.4f}", flush=True)
                if (step+1) % args.save_every == 0 or step+1 == steps:
                    dist.barrier()
                    if rank == 0:
                        path = args.run_dir / f'step{step+1:06d}.pt'
                        save_checkpoint(path, adapter, optimizer, step+1, args,
                                        plan_hash, backbone_hash)
                        print(f'[saved] {path}', flush=True)
                    dist.barrier()
        if rank == 0:
            (args.run_dir/'complete.json').write_text(json.dumps(
                dict(status='complete', steps=steps, history_mode=args.history_mode,
                     plan_sha256=plan_hash,
                     backbone_sha256=backbone_hash, checkpoint=str(args.checkpoint.resolve()),
                     effective_batch=args.accumulation*world_size,
                     final_adapter=str(args.run_dir/f'step{steps:06d}.pt')), indent=2))
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
