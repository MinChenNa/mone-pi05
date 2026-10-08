"""Evaluate the frozen full18 Mone adapter on selected LIBERO-10 tasks only."""

import argparse
import json
from pathlib import Path

import torch

from eval_mone_all_layers import file_sha256, load_adapter
from eval_mone_full_rollout import trial
from eval_mone_heldout import rows_for
from eval_mone_rollout import SUITE_LIMITS
from plan_mone_full import digest
from train_mone_adapter_contrastive import make_policy
from openpi import transforms
from openpi.policies import libero_policy
from openpi.shared import normalize
from openpi.training import config


def plan_task(plan, task_id):
    selected = [(language, meta) for language, meta in plan['task_lookup'].items()
                if meta['suite'] == 'libero_10' and meta['task_id'] == task_id]
    if len(selected) != 1:
        raise ValueError(f'Expected exactly one LIBERO-10 task {task_id}, found {len(selected)}')
    language, _ = selected[0]
    partitions = [name for name, tasks in plan['partitions'].items() if language in tasks]
    if len(partitions) != 1:
        raise ValueError(f'Task {task_id} has invalid plan partition: {partitions}')
    return language, partitions[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--memory-adapter', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--task-id', type=int, choices=(4, 8, 9), required=True)
    parser.add_argument('--mode', choices=('relevant', 'static_initial', 'current_frame', 'same_task'),
                        default='relevant')
    parser.add_argument('--data-dir', type=Path)
    parser.add_argument('--trials', type=int, default=50)
    parser.add_argument('--history-tokens', type=int, default=32)
    parser.add_argument('--wait-steps', type=int, default=10)
    parser.add_argument('--replan-steps', type=int, default=5)
    parser.add_argument('--num-steps', type=int, default=10)
    parser.add_argument('--resize-size', type=int, default=224)
    parser.add_argument('--seed', type=int, default=7)
    args = parser.parse_args()
    args.suite = 'libero_10'
    if min(args.trials, args.history_tokens, args.wait_steps,
           args.replan_steps, args.num_steps) < 1 or args.replan_steps > 10:
        parser.error('Invalid rollout settings')
    if args.mode == 'same_task' and args.data_dir is None:
        parser.error('--data-dir is required for same_task')
    if args.output.exists() or args.output.with_suffix('.jsonl').exists():
        raise FileExistsError(args.output)
    if not torch.cuda.is_available():
        raise RuntimeError('Mone LIBERO-10 rollout requires CUDA')

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    plan = json.loads(args.plan.read_text())
    plan_hash = plan.pop('plan_sha256')
    if digest(plan) != plan_hash:
        raise ValueError('Frozen data plan hash mismatch')
    language, partition = plan_task(plan, args.task_id)
    donor_rows = []
    donor_episode = None
    if args.mode == 'same_task':
        donor_episode = plan['episodes'][language][2]['episode']
        donor_rows = rows_for(args.data_dir, donor_episode)
    device = torch.device('cuda:0')
    model, policy = make_policy(args.checkpoint, device)
    backbone_hash = file_sha256(args.checkpoint/'model.safetensors')
    memory, saved = load_adapter(args.memory_adapter, plan_hash, backbone_hash, device)
    if saved['history_mode'] != 'true_history' or saved['history_tokens'] != args.history_tokens:
        raise ValueError('Wrong Mone adapter or history token count')
    cfg = config.get_config('pi05_libero').model
    norm_stats = normalize.load(args.checkpoint/'assets/physical-intelligence/libero')
    output_transform = transforms.compose([
        *config.ModelTransformFactory()(cfg).outputs,
        transforms.Unnormalize({'actions': norm_stats['actions']}, use_quantiles=True),
        libero_policy.LiberoOutputs(),
    ])
    suite = benchmark.get_benchmark_dict()[args.suite]()
    task = suite.get_task(args.task_id)
    if task.language != language:
        raise ValueError(f'Benchmark task changed: {language} vs {task.language}')
    init_path = Path(get_libero_path('init_states'))/task.problem_folder/task.init_states_file
    init_states = torch.load(init_path, weights_only=False)
    if len(init_states) < args.trials:
        raise ValueError(f'Only {len(init_states)} initial states for {language}')
    bddl = Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    records = []
    try:
        with args.output.with_suffix('.jsonl').open('x') as stream:
            for initial_index in range(args.trials):
                row = trial(args, env, init_states[initial_index], language,
                            args.task_id, initial_index, args.mode, donor_rows, None, None,
                            model, policy, memory, None, output_transform, device)
                records.append(row)
                stream.write(json.dumps(row) + '\n')
                stream.flush()
                successes = sum(item['success'] for item in records)
                print(f'[rollout] task={args.task_id} trial={initial_index} '
                      f'success={row["success"]} steps={row["steps"]} '
                      f'cumulative={successes}/{len(records)}', flush=True)
    finally:
        env.close()
    successes = sum(item['success'] for item in records)
    args.output.write_text(json.dumps(dict(
        status='complete', suite=args.suite, task_id=args.task_id, task=language,
        plan_partition=partition, model='mone_full18_true_history',
        memory_input_mode=args.mode,
        donor_episode=donor_episode,
        checkpoint=str(args.checkpoint.resolve()),
        memory_adapter=str(args.memory_adapter.resolve()),
        backbone_sha256=backbone_hash, plan_sha256=plan_hash,
        trials=len(records), successes=successes,
        success_rate=successes/len(records), seed=args.seed,
        wait_steps=args.wait_steps, replan_steps=args.replan_steps,
        num_steps=args.num_steps, max_steps=SUITE_LIMITS[args.suite],
        pi05_baseline_evaluated=False,
    ), indent=2))
    print(f'[done] task={args.task_id} success={successes}/{len(records)} '
          f'report={args.output}', flush=True)


if __name__ == '__main__':
    main()
