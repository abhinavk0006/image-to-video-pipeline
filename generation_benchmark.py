"""Automatically capture each generation attempt and prepare external video metrics."""
from pathlib import Path
from datetime import datetime, timezone
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from benchmark import collect_run, file_sha256


def run_benchmarked(command, *, knowledge_file, experiment_dir, report_root, cwd, action=None, execute_metrics=True):
    knowledge_file, experiment_dir = Path(knowledge_file), Path(experiment_dir)
    directory = Path(report_root) / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:8])
    directory.mkdir(parents=True)
    knowledge = json.loads(knowledge_file.read_text(encoding='utf-8'))
    raw_paths = [experiment_dir / (clip.get('name') or f'clip_{i+1}') for i, clip in enumerate(knowledge['clips'])]
    raw_paths = [p.with_name(p.name + '.mp4') for p in raw_paths]
    existing = [str(p) for p in raw_paths if p.is_file()]
    versions = {}
    for package in ('torch', 'transformers', 'diffusers', 'imageio-ffmpeg', 'vbench'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    manifest = {'schema_version':1, 'command':list(map(str,command)), 'knowledge_sha256':file_sha256(knowledge_file),
        'generation_status':'running', 'versions':versions, 'existing_clips_before_run':existing,
        'timing_scope':'invocation including startup; may include resumed clips',
        'eligible_for_cold_generation_timing':not bool(existing), 'scientific_accuracy':None,
        'visual_metric_status':'not_prepared', 'errors':[]}
    (directory/'experiment.json').write_text(json.dumps(knowledge,indent=2),encoding='utf-8')
    def save():
        (directory/'run.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    save()
    started=time.perf_counter()
    error=None
    try:
        if action is None:
            environment={**os.environ, 'WAN_BENCHMARK_MANAGED':'1'}
            result=subprocess.run(command,cwd=cwd,check=False,env=environment)
            returncode=result.returncode
        else:
            action()
            returncode=0
        manifest['returncode']=returncode
        manifest['generation_status']='success' if returncode==0 else 'failed'
        if returncode:
            error=subprocess.CalledProcessError(returncode,command)
    except BaseException as exc:
        manifest['generation_status']='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed'
        manifest['errors'].append(f'Generation: {type(exc).__name__}: {exc}')
        error=exc
    finally:
        manifest['pipeline_wall_seconds']=time.perf_counter()-started
        save()
        try:
            collect_run(knowledge_file,experiment_dir,directory,wall_seconds=manifest['pipeline_wall_seconds'],
                        generation_status=manifest['generation_status'])
        except Exception as exc:
            manifest['errors'].append(f'Technical collection: {exc}')
        available=[p for p in raw_paths if p.is_file()]
        manifest['raw_videos']=[{'path':str(p.resolve()),'sha256':file_sha256(p)} for p in available]
        manifest['missing_videos']=[str(p) for p in raw_paths if not p.is_file()]
        if available:
            try:
                evaluate=execute_metrics and bool(shutil.which('vbench'))
                prepared=subprocess.run([sys.executable,str(Path(cwd)/'research_metrics.py'),'--videos',
                    *map(str,available),'--output-dir',str(directory/'vbench'),
                    *(['--execute'] if evaluate else [])],check=False)
                manifest['visual_metric_status']=('scored' if evaluate else 'prepared_not_scored') if prepared.returncode==0 else 'evaluation_failed'
                manifest['visual_metric_note']='Official VBench must be installed to obtain actual quality scores.'
                if prepared.returncode:
                    manifest['errors'].append(f'VBench preparation exited {prepared.returncode}')
            except Exception as exc:
                manifest['visual_metric_status']='preparation_failed'
                manifest['errors'].append(f'VBench preparation: {exc}')
        else:
            manifest['visual_metric_status']='no_generated_videos'
        save()
        print(f'Generation benchmark: {directory}')
        if manifest['errors']:
            print('Benchmark issues:', '; '.join(manifest['errors']))
    if error is not None:
        raise error
    if manifest['errors']:
        raise RuntimeError(f'Generation completed but benchmark capture is incomplete: {directory}')
    return directory, manifest['pipeline_wall_seconds']
