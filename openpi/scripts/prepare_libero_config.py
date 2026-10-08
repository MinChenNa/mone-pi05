"""Create a project-local LIBERO config from a complete upstream checkout."""

import argparse
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--libero-root', type=Path, required=True,
                        help='Full upstream LIBERO checkout, including simulation assets')
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path,
                        default=Path('../assets/libero_config/config.yaml'))
    args = parser.parse_args()
    benchmark = args.libero_root.resolve() / 'libero/libero'
    paths = dict(benchmark_root=benchmark, bddl_files=benchmark / 'bddl_files',
                 init_states=benchmark / 'init_files', assets=benchmark / 'assets',
                 datasets=args.data_dir.resolve())
    missing = [str(path) for path in paths.values() if not path.is_dir()]
    if missing:
        parser.error('Required directories are missing: ' + ', '.join(missing))
    if args.output.exists():
        parser.error(f'Config already exists; choose a new --output: {args.output}')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump({key: str(path) for key, path in paths.items()}),
                           encoding='utf-8')
    print(args.output.resolve())


if __name__ == '__main__':
    main()
