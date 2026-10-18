"""Run official VBench custom-input dimensions without relabelling scores."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import time

from science_evidence import sha256, write_json

CUSTOM_DIMENSIONS = ('subject_consistency', 'background_consistency',
    'motion_smoothness', 'dynamic_degree', 'aesthetic_quality', 'imaging_quality')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--videos', nargs='+', required=True,
        help='Raw generated clips only; exclude title cards and slowed presentation copies.')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--dimensions', nargs='+', choices=CUSTOM_DIMENSIONS, default=list(CUSTOM_DIMENSIONS))
    parser.add_argument('--execute', action='store_true', help='Run installed official vbench; otherwise prepare the exact commands only.')
    args = parser.parse_args()
    root = Path(args.output_dir).resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError('Use a new metric report directory')
    root.mkdir(parents=True, exist_ok=True)
    videos = []
    for index, name in enumerate(args.videos):
        source = Path(name).resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = root/'videos'/f'clip_{index:04d}.mp4'
        destination.parent.mkdir(exist_ok=True)
        shutil.copyfile(source, destination)
        videos.append({'source': str(source), 'sha256': sha256(source), 'input': str(destination)})
    executable = shutil.which('vbench')
    commands = [[executable or 'vbench', 'evaluate', '--dimension', dimension,
        '--videos_path', str(root/'videos'), '--mode=custom_input'] for dimension in args.dimensions]
    try:
        version = importlib.metadata.version('vbench')
    except importlib.metadata.PackageNotFoundError:
        version = None
    report = {'backend': 'official_vbench', 'version': version,
        'protocol': 'custom_input', 'videos': videos, 'commands': commands,
        'status': 'prepared', 'dimensions': {}, 'source': 'https://github.com/Vchitect/VBench',
        'interpretation': 'Official raw scores; not percentages realistic, not scientific accuracy, and not a standard-suite leaderboard total.'}
    write_json(root/'metric_run.json', report)
    if args.execute:
        if not executable:
            raise RuntimeError('Install official VBench in a separate evaluation environment first; commands were saved.')
        for dimension, command in zip(args.dimensions, commands):
            started = time.perf_counter()
            folder = root/dimension
            folder.mkdir()
            with (folder/'log.txt').open('w', encoding='utf-8') as log:
                result = subprocess.run(command, cwd=folder, stdout=log, stderr=subprocess.STDOUT)
            report['dimensions'][dimension] = {'returncode': result.returncode,
                'wall_seconds': time.perf_counter()-started, 'raw_output_directory': str(folder)}
            scores = []
            for path in folder.rglob('*_eval_results.json'):
                payload = json.loads(path.read_text())
                values = payload.get(dimension)
                if isinstance(values, list) and values and isinstance(values[0], (float, int)):
                    scores.append({'file': str(path), 'raw_score': values[0]})
            report['dimensions'][dimension]['raw_scores'] = scores
            report['status'] = 'running'
            write_json(root/'metric_run.json', report)
            if result.returncode:
                report['status'] = 'failed'
                write_json(root/'metric_run.json', report)
                raise RuntimeError(f'Official VBench dimension failed: {dimension}; see {folder / "log.txt"}')
        report['status'] = 'completed'
        write_json(root/'metric_run.json', report)
    print(root/'metric_run.json')


if __name__=='__main__':
    main()
