"""Freeze disjoint LIBERO task splits for all-layer Mone experiments."""

import argparse
import ast
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re

import numpy as np


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def build_plan(data_dir, *, seed=20260927, train_tasks=24, val_tasks=8):
    metadata = [json.loads(line) for line in (data_dir / 'meta/episodes.jsonl').read_text().splitlines()]
    groups = defaultdict(list)
    for item in metadata:
        if item['length'] >= 30:
            groups[item['tasks'][0]].append(dict(episode=item['episode_index'], length=item['length']))
    tasks = set(groups)
    if len(tasks) != 40 or train_tasks != 24 or val_tasks != 8:
        raise ValueError('Expected four LIBERO suites with 10 tasks each and 24/8/8 split')
    mapping_path = (Path(__file__).resolve().parents[2] / 'vendor/libero/libero/libero'
                    / 'benchmark/libero_suite_task_map.py')
    syntax = ast.parse(mapping_path.read_text())
    mapping = ast.literal_eval(syntax.body[0].value)
    suites = ('libero_spatial', 'libero_object', 'libero_goal', 'libero_10')
    lookup = {}
    for suite in suites:
        for task_id, name in enumerate(mapping[suite]):
            language = re.sub(r'^.*?_SCENE\d+_', '', name).replace('_', ' ')
            if language in lookup:
                raise ValueError(f'Duplicate LIBERO language: {language}')
            lookup[language] = dict(suite=suite, task_id=task_id, benchmark_name=name)
    if set(lookup) != tasks:
        raise ValueError(f'Suite/data task mismatch: missing={tasks-set(lookup)} extra={set(lookup)-tasks}')
    rng = np.random.default_rng(seed)
    partitions = dict(train=[], val=[], test=[])
    for suite in suites:
        order = rng.permutation(sorted(task for task in tasks if lookup[task]['suite'] == suite)).tolist()
        partitions['train'].extend(order[:6])
        partitions['val'].extend(order[6:8])
        partitions['test'].extend(order[8:])
    for task in tasks:
        if len(groups[task]) < 3:
            raise ValueError(f'Need >=3 episodes for task {task}')
    result = dict(status='prepared', seed=seed, partitions=partitions, task_lookup=lookup,
                  episodes={task: sorted(groups[task], key=lambda item: item['episode']) for task in tasks},
                  history_lag=10, action_horizon=10,
                  note='Tasks are disjoint across train, val and final test. No previous adapter is reused.')
    result['plan_sha256'] = digest(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=20260927)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = build_plan(args.data_dir, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(f"[plan] train={len(result['partitions']['train'])} "
          f"val={len(result['partitions']['val'])} test={len(result['partitions']['test'])} "
          f"sha256={result['plan_sha256']} output={args.output}")


if __name__ == '__main__':
    main()
