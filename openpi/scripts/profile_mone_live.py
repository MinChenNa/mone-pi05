"""Read-only 60-second profile of an already running Mone training job.

Run on the GPU node. This does not attach to or modify the training process.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import subprocess
import time


GPU_FIELDS = 'index,utilization.gpu,utilization.memory,memory.used,power.draw'


def gpu_sample():
    result = subprocess.run(
        ['nvidia-smi', f'--query-gpu={GPU_FIELDS}', '--format=csv,noheader,nounits'],
        capture_output=True, text=True, check=True, timeout=10,
    )
    rows = {}
    for line in result.stdout.splitlines():
        fields = [item.strip() for item in line.split(',')]
        if len(fields) != 5:
            raise ValueError(f'Unexpected nvidia-smi row: {line}')
        index = int(fields[0])
        rows[index] = dict(gpu_percent=float(fields[1]),
                           memory_percent=float(fields[2]),
                           memory_mib=float(fields[3]),
                           power_w=float(fields[4]))
    if len(rows) != 4:
        raise RuntimeError(f'Expected four visible GPUs, found {len(rows)}')
    return rows


def workers(run_dir):
    matches = {}
    target = str(run_dir)
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            cmdline = (entry / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
            if 'train_mone_all_layers.py' in cmdline and target in cmdline:
                matches[int(entry.name)] = entry
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
    return matches


def process_stats(path):
    stat = (path / 'stat').read_text().rsplit(')', 1)[1].split()
    io = {}
    for line in (path / 'io').read_text().splitlines():
        key, value = line.split(':', 1)
        io[key] = int(value.strip())
    rss_kib = 0
    for line in (path / 'status').read_text().splitlines():
        if line.startswith('VmRSS:'):
            rss_kib = int(line.split()[1])
            break
    return dict(cpu_ticks=int(stat[11]) + int(stat[12]),
                read_bytes=io.get('read_bytes', 0),
                read_chars=io.get('rchar', 0),
                rss_mib=rss_kib / 1024)


def system_stats():
    numbers = [int(item) for item in Path('/proc/stat').read_text().splitlines()[0].split()[1:]]
    available_kib = next(int(line.split()[1]) for line in
                         Path('/proc/meminfo').read_text().splitlines()
                         if line.startswith('MemAvailable:'))
    return dict(total_ticks=sum(numbers), iowait_ticks=numbers[4],
                available_gib=available_kib / 1024 / 1024)


def last_step(metrics):
    if not metrics.exists():
        return None
    with metrics.open('rb') as stream:
        stream.seek(0, 2)
        end = stream.tell()
        stream.seek(max(0, end - 16384))
        lines = stream.read().splitlines()
    for line in reversed(lines):
        try:
            return int(json.loads(line)['step'])
        except (ValueError, KeyError):
            continue
    return None


def summarize(samples, start_step, end_step, elapsed):
    report = dict(elapsed_seconds=round(elapsed, 2), start_step=start_step,
                  end_step=end_step,
                  steps_per_minute=(None if start_step is None or end_step is None else
                                    round((end_step - start_step) * 60 / elapsed, 2)),
                  sampled_worker_counts=[len(item['workers']) for item in samples],
                  gpu={})
    for index in range(4):
        rows = [item['gpu'][index] for item in samples]
        busy = [row['gpu_percent'] for row in rows]
        report['gpu'][str(index)] = dict(
            mean_busy_percent=round(statistics.mean(busy), 1),
            median_busy_percent=round(statistics.median(busy), 1),
            idle_sample_fraction=round(sum(value == 0 for value in busy) / len(busy), 2),
            mean_memory_bandwidth_percent=round(statistics.mean(
                row['memory_percent'] for row in rows), 1),
            mean_allocated_mib=round(statistics.mean(row['memory_mib'] for row in rows)),
            mean_power_w=round(statistics.mean(row['power_w'] for row in rows), 1),
        )
    first, last = samples[0], samples[-1]
    common = set(first['workers']) & set(last['workers'])
    ticks = sum(last['workers'][pid]['cpu_ticks'] - first['workers'][pid]['cpu_ticks']
                for pid in common)
    read_bytes = sum(last['workers'][pid]['read_bytes'] - first['workers'][pid]['read_bytes']
                     for pid in common)
    read_chars = sum(last['workers'][pid]['read_chars'] - first['workers'][pid]['read_chars']
                     for pid in common)
    import os
    report['worker_cpu_cores_used'] = round(ticks / os.sysconf('SC_CLK_TCK') / elapsed, 2)
    report['worker_physical_read_mib_per_sec'] = round(read_bytes / 1048576 / elapsed, 2)
    report['worker_read_calls_mib_per_sec'] = round(read_chars / 1048576 / elapsed, 2)
    report['worker_rss_gib_last'] = round(sum(
        value['rss_mib'] for value in last['workers'].values()) / 1024, 2)
    report['host_available_gib_last'] = round(last['system']['available_gib'], 2)
    total_delta = last['system']['total_ticks'] - first['system']['total_ticks']
    report['host_iowait_percent'] = round(100 * (
        last['system']['iowait_ticks'] - first['system']['iowait_ticks']) / total_delta, 2)
    report['note'] = ('External counters identify underfeeding but cannot attribute exact '
                      'time to Parquet decode versus memory kernels; that needs stage timing.')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=Path('../outputs/full18/full18_20260927/memory'))
    parser.add_argument('--seconds', type=int, default=60)
    parser.add_argument('--interval', type=float, default=1.0)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.seconds < 10 or args.interval <= 0:
        parser.error('Need at least 10 seconds and a positive interval')
    if not workers(args.run_dir):
        raise SystemExit(f'No live training workers found for --run-dir {args.run_dir}; run on the H200 node')
    attempts = list(args.run_dir.glob('rank0*.jsonl'))
    metrics = max(attempts, key=lambda path: path.stat().st_mtime) if attempts else args.run_dir / 'rank0.jsonl'
    start_step = last_step(metrics)
    start = time.monotonic()
    samples = []
    while True:
        now = time.monotonic()
        active = {}
        for pid, path in workers(args.run_dir).items():
            try:
                active[pid] = process_stats(path)
            except (FileNotFoundError, PermissionError, ProcessLookupError):
                continue
        samples.append(dict(elapsed=round(now - start, 2), gpu=gpu_sample(),
                            workers=active, system=system_stats()))
        if time.monotonic() - start >= args.seconds:
            break
        time.sleep(max(0, args.interval - (time.monotonic() - now)))
    elapsed = time.monotonic() - start
    result = summarize(samples, start_step, last_step(metrics), elapsed)
    output = args.output or Path('../outputs/full18') / (
        'live-profile-' + datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S') + '.json')
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f'[profile] report={output}')


if __name__ == '__main__':
    main()
