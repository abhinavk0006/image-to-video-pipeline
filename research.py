"""Research tools: evidence, causal pairs, challenge footage and score reports."""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import time

from knowledge_loader import load_experiment_from_json
from science_evidence import (write_json, review_clip, repair_plan,
                              visibility_preflight, choose_profile, sha256, contract_for)


def make_causal_pair(experiment, pointer, value, expected, directory):
    """One manipulated input; expected outcomes are stored outside generator inputs."""
    base = json.loads(Path(experiment).read_text())
    variant = copy.deepcopy(base)
    parts = pointer.strip('/').split('/')
    target = variant
    for part in parts[:-1]:
        target = target[int(part)] if isinstance(target, list) else target[part]
    key = int(parts[-1]) if isinstance(target, list) else parts[-1]
    original = copy.deepcopy(target[key])
    if original == value:
        raise ValueError('A causal pair must change a condition')
    target[key] = value
    directory = Path(directory)
    if directory.exists() and any(directory.iterdir()):
        raise ValueError('Use an empty causal-pair directory')
    write_json(directory / 'control.json', base)
    write_json(directory / 'intervention.json', variant)
    return write_json(directory / 'pair.json', {'schema_version': 1,
        'changed_input': pointer, 'control_value': original, 'intervention_value': value,
        'expected_outcomes': expected, 'source_reviewed': False, 'reviewer': '',
        'source_ids': [], 'status': 'draft',
        'review_checklist': ['Verify the intervention and both expected outcomes against sources.',
            'Check that unchanged prompts/references do not contradict the changed condition.',
            'Keep expected evaluation outcomes separate from verifier input.',
            'Use the same seeds, generator and compute budget for both conditions.']})


def make_challenge(kind, videos, output):
    """Create real edited footage; semantic ground truth still needs review."""
    from stitcher import stitch_clips
    from imageio_ffmpeg import get_ffmpeg_exe
    from science_evidence import video_info
    paths = [Path(p).resolve() for p in videos]
    output = Path(output).resolve()
    if output.exists() or output in paths:
        raise ValueError('Challenge output must be new and distinct from source videos')
    if kind == 'wrong_order':
        if len(paths) < 2:
            raise ValueError('Wrong-order challenge needs two or more ordered source clips')
        stitch_clips(list(reversed(paths)), output)
    elif kind == 'omitted_action':
        if len(paths) < 2:
            raise ValueError('Omission challenge needs at least two ordered source clips')
        stitch_clips(paths[1:], output)
    elif kind == 'premature_observation':
        if len(paths) != 1:
            raise ValueError('Premature-observation challenge needs one reviewed source clip')
        info = video_info(paths[0])
        split = max(1, info['frames']//2)
        filters = (f'[0:v]split[a][b];[a]trim=start_frame={split},setpts=PTS-STARTPTS[x];'
                   f'[b]trim=end_frame={split},setpts=PTS-STARTPTS[y];[x][y]concat=n=2:v=1:a=0[v]')
        output.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([get_ffmpeg_exe(), '-n', '-i', str(paths[0]), '-filter_complex', filters,
            '-map', '[v]', '-an', '-c:v', 'libx264', '-crf', '18', str(output)], check=True, capture_output=True)
    else:
        raise ValueError('Unsupported challenge kind')
    return write_json(output.with_suffix('.challenge.json'), {'kind': kind,
        'sources': [{'path': str(p), 'sha256': sha256(p)} for p in paths],
        'output_sha256': sha256(output), 'expected_valid': None, 'reviewer': '',
        'ground_truth_reviewed': False,
        'note': 'Review footage before labelling invalid. An edit alone does not establish a scientific violation.'})


def summarize_evaluations(records, *, unit='sequence'):
    """Percentages with explicit denominators; pending and missing stay visible."""
    rows = records
    accepted = sum(r.get('decision') == 'accepted' for r in rows)
    decided = sum(r.get('decision') in {'accepted', 'rejected'} for r in rows)
    labelled = [r for r in rows if r.get('ground_truth_reviewed') is True and isinstance(r.get('expected_valid'), bool)]
    negatives = [r for r in labelled if r['expected_valid'] is False]
    false_accepts = sum(r.get('decision') == 'accepted' for r in negatives)
    judged = [r for r in labelled if r.get('decision') in {'accepted', 'rejected'}]
    correct = sum((r['decision'] == 'accepted') == r['expected_valid'] for r in judged)
    def fraction(numerator, denominator):
        if not denominator:
            return {'count': numerator, 'denominator': denominator, 'percent': None, 'wilson_95_percent': None}
        p, z = numerator/denominator, 1.96
        denom = 1+z*z/denominator
        center = (p+z*z/(2*denominator))/denom
        radius = z*math.sqrt(p*(1-p)/denominator+z*z/(4*denominator**2))/denom
        return {'count': numerator, 'denominator': denominator, 'percent': 100*p,
                'wilson_95_percent': [100*(center-radius), 100*(center+radius)]}
    return {'unit': unit, 'units': len(rows), 'acceptance_all_units': fraction(accepted, len(rows)),
        'decision_coverage': fraction(decided, len(rows)),
        'false_acceptance_all_reviewed_invalid': fraction(false_accepts, len(negatives)),
        'accuracy_on_decided_labelled': fraction(correct, len(judged)),
        'invalid_decision_coverage': fraction(sum(r.get('decision') in {'accepted', 'rejected'} for r in negatives), len(negatives)),
        'limitations': 'Intervals assume independent samples; aggregate correlated clips by experiment and use experiment-level resampling for paper results.'}


def blind_manifest(manifest, directory, seed):
    """Separate private method/ground-truth keys from reviewer assignments."""
    data = json.loads(Path(manifest).read_text())
    rows = list(data['videos'])
    random.Random(seed).shuffle(rows)
    directory = Path(directory)
    if directory.exists() and any(directory.iterdir()):
        raise ValueError('Use an empty blinded-review directory')
    private, public = [], []
    for index, row in enumerate(rows):
        import shutil
        identifier = f'video_{index+1:04d}'
        source = Path(row['path']).resolve()
        destination = directory / 'videos' / (identifier+'.mp4')
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        private.append({'id': identifier, **row, 'sha256': sha256(source)})
        public.append({'id': identifier, 'path': 'videos/'+identifier+'.mp4',
                       'realism_rating_1_to_5': None, 'scientific_valid': None,
                       'reviewer': '', 'notes': ''})
    write_json(directory / 'private_key.json', {'seed': seed, 'videos': private})
    return write_json(directory / 'reviewer_manifest.json', {'videos': public,
        'instructions': 'Share only this manifest and videos, not private_key.json. Use the same independent scientific rubric for every method.'})


def unblind(private_key, reviews):
    key = {row['id']: row for row in json.loads(Path(private_key).read_text())['videos']}
    observations = json.loads(Path(reviews).read_text())['videos']
    if len({row['id'] for row in observations}) != len(observations):
        raise ValueError('Duplicate blinded review IDs')
    observations = {row['id']: row for row in observations}
    if set(observations) - set(key):
        raise ValueError('Review ID is absent from the private assignment key')
    records = []
    for identifier, original in key.items():
        row = observations.get(identifier, {'id': identifier})
        valid = row.get('scientific_valid')
        if valid is not None and not isinstance(valid, bool):
            raise ValueError('scientific_valid must be true, false or null')
        rating = row.get('realism_rating_1_to_5')
        if rating is not None and (not isinstance(rating, (int, float)) or not math.isfinite(rating) or not 1 <= rating <= 5):
            raise ValueError('Realism rating must lie within [1,5]')
        reviewed = bool(row.get('reviewer') and row.get('notes'))
        records.append({**original, 'reviewer': row.get('reviewer'), 'notes': row.get('notes'),
            'realism_rating_1_to_5': rating if reviewed else None,
            'decision': 'pending' if valid is None or not reviewed else 'accepted' if valid else 'rejected'})
    groups = {}
    for method in sorted({str(row.get('method', 'unspecified')) for row in records}):
        selected = [r for r in records if str(r.get('method', 'unspecified')) == method]
        ratings = [r['realism_rating_1_to_5'] for r in selected if r['realism_rating_1_to_5'] is not None]
        groups[method] = {**summarize_evaluations(selected, unit='video'),
            'human_realism_mean_1_to_5': sum(ratings)/len(ratings) if ratings else None,
            'human_realism_ratings_n': len(ratings)}
    return {'records': records, 'by_method': groups,
            'limits': 'One review file; inter-rater agreement and adjudication must be assessed separately.'}


def study_assign(participants, directory, seed):
    """Randomized assignment tooling; recruitment, consent and study remain human work."""
    records = json.loads(Path(participants).read_text())
    if any(not p.get('pseudonymous_id') or p.get('consent_recorded') is not True for p in records):
        raise ValueError('Every participant needs a pseudonymous ID and recorded consent')
    if len({p['pseudonymous_id'] for p in records}) != len(records):
        raise ValueError('Participant IDs must be unique')
    random.Random(seed).shuffle(records)
    groups = ('text_diagrams', 'generated_video', 'human_demonstration')
    assignments = [{'pseudonymous_id': p['pseudonymous_id'], 'group': groups[i % 3]}
                   for i, p in enumerate(records)]
    return write_json(Path(directory) / 'private_assignments.json', {'seed': seed,
        'assignments': assignments, 'note': 'Counterbalance experiment difficulty and prior knowledge in the final protocol; assignment alone is not a completed study.'})


def study_scores(path):
    with Path(path).open(encoding='utf-8', newline='') as handle:
        rows = list(csv.DictReader(handle))
    groups = {}
    seen = set()
    for row in rows:
        identity = row['pseudonymous_id']
        if identity in seen:
            raise ValueError('Duplicate participant score')
        seen.add(identity)
        maximum = float(row['maximum_score'])
        if not math.isfinite(maximum) or maximum <= 0:
            raise ValueError('maximum_score must be positive')
        scores = {}
        for key in ('pre', 'post', 'retention'):
            value = None if not row.get(key, '').strip() else float(row[key])
            if value is not None and (not math.isfinite(value) or not 0 <= value <= maximum):
                raise ValueError('Study scores must lie within [0, maximum_score]')
            scores[key] = None if value is None else 100*value/maximum
        scores['gain'] = None if scores['pre'] is None or scores['post'] is None else scores['post']-scores['pre']
        groups.setdefault(row['group'], []).append(scores)
    return {group: {key: {'n': sum(s[key] is not None for s in items),
            'mean_percentage_points': (sum(s[key] for s in items if s[key] is not None)/sum(s[key] is not None for s in items))
                if any(s[key] is not None for s in items) else None}
        for key in ('pre', 'post', 'retention', 'gain')} for group, items in groups.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    review = commands.add_parser('review')
    review.add_argument('--knowledge-file', required=True)
    review.add_argument('--run-dir', required=True)
    review.add_argument('--verifier-script')
    repair = commands.add_parser('repair-plan')
    repair.add_argument('--knowledge-file', required=True)
    repair.add_argument('--failed-clip', required=True)
    repair.add_argument('--continuity-mode', choices=['independent', 'chain'], default='independent')
    repair.add_argument('--output', required=True)
    repair.add_argument('--run-dir')
    repair.add_argument('--apply', action='store_true', help='Archive affected videos and manifests so a normal pipeline rerun regenerates them.')
    pair = commands.add_parser('causal-pair')
    pair.add_argument('--knowledge-file', required=True)
    pair.add_argument('--pointer', required=True)
    pair.add_argument('--value-json', required=True)
    pair.add_argument('--expected-json', required=True)
    pair.add_argument('--output-dir', required=True)
    challenge = commands.add_parser('challenge')
    challenge.add_argument('--kind', choices=['wrong_order', 'omitted_action', 'premature_observation'], required=True)
    challenge.add_argument('--videos', nargs='+', required=True)
    challenge.add_argument('--output', required=True)
    summary = commands.add_parser('summarize')
    summary.add_argument('--records', required=True)
    summary.add_argument('--output', required=True)
    blind = commands.add_parser('blind')
    blind.add_argument('--manifest', required=True)
    blind.add_argument('--output-dir', required=True)
    blind.add_argument('--seed', type=int, default=42)
    unblind_parser = commands.add_parser('unblind')
    unblind_parser.add_argument('--private-key', required=True)
    unblind_parser.add_argument('--reviews', required=True)
    unblind_parser.add_argument('--output', required=True)
    study = commands.add_parser('study-assign')
    study.add_argument('--participants', required=True)
    study.add_argument('--output-dir', required=True)
    study.add_argument('--seed', type=int, default=42)
    scoring = commands.add_parser('study-scores')
    scoring.add_argument('--scores', required=True)
    scoring.add_argument('--output', required=True)
    profiles = commands.add_parser('select-profile')
    profiles.add_argument('--measurements', required=True)
    profiles.add_argument('--seconds', type=float, required=True)
    profiles.add_argument('--minimum-area', type=int, required=True)
    profiles.add_argument('--gpu-limit-gib', type=float, action='append', required=True)
    profiles.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.command == 'review':
        experiment = load_experiment_from_json(args.knowledge_file)
        rows = []
        requirement_results = []
        for clip in experiment.clips:
            video = Path(args.run_dir) / (clip.name+'.mp4')
            if not video.is_file():
                rows.append({'clip': clip.name, 'decision': 'missing'})
                requirement_results.extend('unknown' for _ in contract_for(clip)['requirements'])
                continue
            result, review = review_clip(clip, video, Path(args.run_dir)/clip.name/'science', args.verifier_script)
            rows.append({'clip': clip.name, 'decision': result})
            requirement_results.extend(review['requirement_results'].values())
        confirmed = sum(value=='pass' for value in requirement_results)
        evaluated = sum(value in {'pass', 'fail'} for value in requirement_results)
        sequence_decision = 'accepted' if rows and all(r['decision']=='accepted' for r in rows) else (
            'rejected' if any(r['decision']=='rejected' for r in rows) else 'pending')
        write_json(Path(args.run_dir)/'scientific_summary.json', {'clips': rows, **summarize_evaluations(rows, unit='clip'),
            'sequence_decision': sequence_decision,
            'scientific_requirements': {'total': len(requirement_results), 'confirmed_pass': confirmed,
                'evaluated': evaluated, 'unknown': len(requirement_results)-evaluated,
                'pass_percent_all': 100*confirmed/len(requirement_results) if requirement_results else None,
                'pass_percent_evaluated': 100*confirmed/evaluated if evaluated else None,
                'coverage_percent': 100*evaluated/len(requirement_results) if requirement_results else None}})
    elif args.command == 'repair-plan':
        experiment = load_experiment_from_json(args.knowledge_file)
        plan = repair_plan(experiment, args.failed_clip, continuity_mode=args.continuity_mode)
        write_json(args.output, plan)
        if args.apply:
            if not args.run_dir:
                parser.error('--apply requires --run-dir')
            root = Path(args.run_dir).resolve()
            for name in plan['invalidate']:
                for suffix in ('.mp4', '.generation.json'):
                    source = (root / (name+suffix)).resolve()
                    if source.parent != root:
                        raise ValueError('Invalid clip path')
                    if source.exists():
                        source.rename(source.with_name(source.name+f'.invalidated-{time.time_ns()}'))
    elif args.command == 'causal-pair':
        make_causal_pair(args.knowledge_file, args.pointer, json.loads(args.value_json), json.loads(args.expected_json), args.output_dir)
    elif args.command == 'challenge':
        make_challenge(args.kind, args.videos, args.output)
    elif args.command == 'summarize':
        write_json(args.output, summarize_evaluations(json.loads(Path(args.records).read_text())))
    elif args.command == 'blind':
        blind_manifest(args.manifest, args.output_dir, args.seed)
    elif args.command == 'unblind':
        write_json(args.output, unblind(args.private_key, args.reviews))
    elif args.command == 'study-assign':
        study_assign(args.participants, args.output_dir, args.seed)
    elif args.command == 'study-scores':
        write_json(args.output, study_scores(args.scores))
    elif args.command == 'select-profile':
        selected = choose_profile(json.loads(Path(args.measurements).read_text()), duration=args.seconds,
            minimum_area=args.minimum_area, gpu_limits=[g*2**30 for g in args.gpu_limit_gib])
        write_json(args.output, {'selected': selected, 'status': 'measured_candidate' if selected else 'no_measured_configuration'})


if __name__ == '__main__':
    main()
