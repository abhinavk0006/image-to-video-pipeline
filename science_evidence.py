"""Scientific contracts and auditable review; expected text is never evidence.

The built-in reviewer is human-backed. No visual-language model is assumed to
be installed, and pixel sharpness cannot establish scientific correctness.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import html
import json
import math
from pathlib import Path
import subprocess
import sys


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')
    return path


def contract_for(clip):
    supplied = dict(clip.metadata.get('science_spec', {}))
    if any(not isinstance(item, str) for item in supplied.get('preconditions', [])):
        raise ValueError('Preconditions must be source-grounded text strings')
    entities = supplied.get('entities', [])
    if any(not isinstance(e, dict) or not e.get('id') or not e.get('description') for e in entities):
        raise ValueError('Entities need persistent IDs and descriptions')
    if len({e['id'] for e in entities}) != len(entities):
        raise ValueError('Entity IDs must be unique')
    requirements = supplied.get('requirements', [])
    if not requirements:
        # Legacy descriptions create draft requirements, never verified facts.
        requirements = [{'id': key, 'description': text, 'critical': True,
                         'evidence_kind': kind}
                        for key, text, kind in (
                            ('action', clip.action, 'interval'),
                            ('observation', clip.observable_target or clip.state_after, 'interval'),
                            ('preservation', '; '.join(clip.preserve_objects), 'interval')) if text]
    ids = set()
    for item in requirements:
        if not isinstance(item, dict) or not item.get('id') or not item.get('description'):
            raise ValueError('Each scientific requirement needs id and description')
        if item['id'] in ids:
            raise ValueError('Scientific requirement IDs must be unique within a clip')
        if item.get('evidence_kind', 'interval') not in {'interval', 'frame'}:
            raise ValueError('evidence_kind must be interval or frame')
        if not isinstance(item.get('critical', True), bool):
            raise ValueError('critical must be boolean')
        ids.add(item['id'])
    temporal = supplied.get('temporal_order', [])
    for before, after in temporal:
        if before not in ids or after not in ids or before == after:
            raise ValueError('Temporal rules must reference two distinct requirements')
    return {'schema_version': 1, 'clip': clip.name,
            'source_ids': clip.metadata.get('source_ids', []),
            'source_reviewed': supplied.get('source_reviewed', False) is True,
            'entities': supplied.get('entities', []),
            'preconditions': supplied.get('preconditions', []),
            'requirements': requirements, 'temporal_order': temporal,
            'intended_state_after': clip.state_after,
            'source_status': clip.metadata.get('source_status', 'unspecified')}


def visibility_preflight(image_path, annotations, *, output_size=None):
    """Measure annotated image regions. Missing annotations remain unknown."""
    from PIL import Image, ImageStat
    with Image.open(image_path) as image:
        width, height = image.size
        target_width, target_height = output_size or image.size
        targets = []
        for target in annotations:
            box = target.get('bbox')
            if not box or len(box) != 4 or any(not math.isfinite(float(v)) for v in box):
                raise ValueError('Visibility bbox needs four finite normalized coordinates')
            x0, y0, x1, y1 = map(float, box)
            if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
                raise ValueError('Visibility bbox must lie within [0,1]')
            projected = min((x1-x0)*target_width, (y1-y0)*target_height)
            crop = image.crop((int(x0*width), int(y0*height), max(int(x1*width), int(x0*width)+1),
                               max(int(y1*height), int(y0*height)+1))).convert('L')
            minimum = float(target.get('minimum_pixels', 24))
            if not math.isfinite(minimum) or minimum <= 0:
                raise ValueError('minimum_pixels must be finite and positive')
            reasons = []
            if projected < minimum:
                reasons.append('target_too_small')
            occluded = target.get('occluded')
            if occluded is not None and not isinstance(occluded, bool):
                raise ValueError('occluded must be true, false or null')
            if occluded is True:
                reasons.append('annotated_occlusion')
            targets.append({'id': target.get('id'), 'projected_min_pixels': projected,
                'region_luminance_stddev': ImageStat.Stat(crop).stddev[0],
                'status': 'fail' if reasons else 'unknown' if occluded is None else 'pass',
                'reasons': reasons})
    return {'image': str(image_path), 'image_sha256': sha256(image_path),
            'measured_size': [width, height], 'projected_size': [target_width, target_height],
            'targets': targets, 'status': 'fail' if any(t['status']=='fail' for t in targets)
            else 'unknown' if not targets or any(t['status']=='unknown' for t in targets) else 'pass',
            'limits': 'Annotated visibility only; luminance variation does not prove contrast, identity or scientific accuracy.'}


def decision(contract, review, video_path):
    if review.get('video_sha256') != sha256(video_path):
        return 'stale'
    if review.get('contract_sha256') != hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest():
        return 'stale'
    if not contract['requirements'] or not contract['source_reviewed'] or not contract['source_ids']:
        return 'pending'
    records = review.get('requirements', {})
    failed, pending = False, not any(r.get('critical', True) for r in contract['requirements'])
    validated_passes = set()
    for required in contract['requirements']:
        item = records.get(required['id'], {})
        status = item.get('status', 'unknown')
        if status not in {'pass', 'fail', 'unknown'}:
            raise ValueError('Evidence status must be pass, fail or unknown')
        intervals = item.get('intervals', [])
        valid = bool(intervals) and bool(item.get('reviewer')) and bool(item.get('notes'))
        for interval in intervals:
            if len(interval) != 2 or any(not math.isfinite(float(t)) for t in interval):
                raise ValueError('Evidence intervals need two finite timestamps')
            start, end = map(float, interval)
            if start < 0 or end < start or end > float(review['duration_seconds']):
                raise ValueError('Evidence interval is outside the decoded video')
            if required.get('evidence_kind', 'interval') == 'interval' and start == end:
                valid = False
        if required.get('critical', True):
            failed |= status == 'fail' and valid
            pending |= status == 'unknown' or (status in {'pass', 'fail'} and not valid)
        if status == 'pass' and valid:
            validated_passes.add(required['id'])
    for before, after in contract['temporal_order']:
        first, second = records.get(before, {}), records.get(after, {})
        if before in validated_passes and after in validated_passes:
            # First observed completion must precede the later event's onset.
            failed |= min(t[1] for t in first['intervals']) > min(t[0] for t in second['intervals'])
        else:
            pending = True
    return 'rejected' if failed else 'pending' if pending else 'accepted'


def video_info(video_path):
    from video_decode import read_rgb_frames
    reader = read_rgb_frames(video_path)
    try:
        metadata = next(reader)
        count = sum(1 for _ in reader)
        fps = float(metadata.get('fps') or 0)
        if not count or not math.isfinite(fps) or fps <= 0:
            raise ValueError('Video has no valid decoded frames/fps')
        return {'fps': fps, 'frames': count, 'duration_seconds': count/fps,
                'last_timestamp_seconds': (count-1)/fps, 'size': list(metadata['size'])}
    finally:
        reader.close()


def review_clip(clip, video_path, directory, verifier_script=None):
    directory = Path(directory)
    contract = contract_for(clip)
    contract_path = write_json(directory / 'contract.json', contract)
    video_hash = sha256(video_path)
    contract_hash = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    path = directory / 'review.json'
    info = video_info(video_path)
    try:
        review = json.loads(path.read_text())
    except (OSError, ValueError):
        review = {}
    if review.get('video_sha256') != video_hash or review.get('contract_sha256') != contract_hash:
        if path.exists():
            import time
            path.rename(directory / f'review.previous-{time.time_ns()}.json')
        review = {'video': str(Path(video_path).resolve()), 'video_sha256': video_hash,
            'contract_sha256': contract_hash, **info, 'requirements': {
                r['id']: {'status': 'unknown', 'intervals': [], 'reviewer': '', 'notes': ''}
                for r in contract['requirements']}, 'handoff_candidates': []}
        write_json(path, review)
    if verifier_script:
        subprocess.run([sys.executable, str(verifier_script), '--video', str(video_path),
            '--contract', str(contract_path), '--review', str(path)], check=True, timeout=300)
        review = json.loads(path.read_text())
    # Neither manually edited reviews nor plugins define the decoded duration.
    review.update(info)
    review['video'] = str(Path(video_path).resolve())
    result = decision(contract, review, video_path)
    review['decision'] = result
    review['requirement_results'] = {}
    grounded = contract['source_reviewed'] and bool(contract['source_ids']) and result != 'stale'
    for requirement in contract['requirements']:
        item = review['requirements'].get(requirement['id'], {})
        intervals = item.get('intervals', [])
        justified = bool(item.get('reviewer') and item.get('notes') and intervals)
        if requirement.get('evidence_kind', 'interval') == 'interval':
            justified = justified and all(pair[1] > pair[0] for pair in intervals)
        review['requirement_results'][requirement['id']] = item.get('status', 'unknown') if grounded and justified else 'unknown'
    write_json(path, review)
    render_review_html(directory, contract, review)
    return result, review


def render_review_html(directory, contract, review):
    directory = Path(directory)
    import os
    video = Path(review['video'])
    relative = Path(os.path.relpath(video, directory)).as_posix()
    rows = []
    for required in contract['requirements']:
        item = review['requirements'].get(required['id'], {})
        links = ' '.join(f'<button data-time="{float(pair[0])}">{float(pair[0]):.2f}s–{float(pair[1]):.2f}s</button>'
                         for pair in item.get('intervals', []))
        rows.append('<tr><td>'+html.escape(required['description'])+'</td><td>'+
            html.escape(item.get('status', 'unknown'))+'</td><td>'+links+'</td><td>'+
            html.escape(item.get('notes', ''))+'</td></tr>')
    body = '<!doctype html><meta charset="utf-8"><title>Scientific evidence review</title><style>body{font:16px system-ui;max-width:1000px;margin:30px auto}video{width:100%;max-height:500px}td,th{padding:12px;border:1px solid #ccc}table{border-collapse:collapse}</style>'
    body += '<h1>'+html.escape(contract['clip'])+'</h1><p>Decision: '+html.escape(review['decision'])+'</p>'
    body += '<p>Edit review.json to record observations. Expected conditions are not observed evidence.</p>'
    body += '<video id="footage" controls src="'+html.escape(relative, quote=True)+'"></video>'
    body += '<table><tr><th>Requirement</th><th>Status</th><th>Evidence</th><th>Notes</th></tr>'+''.join(rows)+'</table>'
    body += '<script>document.querySelectorAll("[data-time]").forEach(b=>b.onclick=()=>{let v=document.getElementById("footage");v.currentTime=Number(b.dataset.time);v.play()})</script>'
    (directory / 'review.html').write_text(body, encoding='utf-8')


def select_reviewed_handoff(video, review_path, required_ids, output):
    """Only use explicitly reviewed candidates covering the next prerequisites."""
    if not required_ids:
        raise ValueError('Reviewed handoff needs explicit producer requirement IDs')
    review = json.loads(Path(review_path).read_text())
    if review.get('video_sha256') != sha256(video):
        raise ValueError('Handoff review is stale')
    contract_path = Path(review_path).with_name('contract.json')
    contract = json.loads(contract_path.read_text())
    if decision(contract, review, video) != 'accepted':
        raise ValueError('Producer scientific review must be accepted before selecting its handoff')
    if not set(required_ids).issubset({r['id'] for r in contract['requirements']}):
        raise ValueError('Handoff references unknown producer requirements')
    info = video_info(video)
    candidates = []
    for candidate in review.get('handoff_candidates', []):
        timestamp = float(candidate['timestamp_seconds'])
        if not math.isfinite(timestamp) or not 0 <= timestamp <= info['last_timestamp_seconds']:
            raise ValueError('Handoff timestamp is outside the decoded video')
        if not candidate.get('reviewer') or not candidate.get('notes'):
            continue
        if all(candidate.get('conditions', {}).get(key) == 'pass' for key in required_ids):
            candidates.append(candidate)
    if not candidates:
        raise ValueError('No reviewed frame satisfies the next action preconditions')
    selected = max(candidates, key=lambda c: float(c['timestamp_seconds']))
    from frame_extractor import FrameExtractor
    return FrameExtractor().extract_handoff_frame(video, output, mode='offset',
        offset_seconds=info['last_timestamp_seconds']-float(selected['timestamp_seconds']))


def dependency_map(experiment, continuity_mode='independent'):
    from pipeline import clip_requires_continuity
    result = {}
    for index, clip in enumerate(experiment.clips):
        own_reference = bool(clip.input_frame_path or clip.generated_image_path)
        parent = '' if own_reference or not clip_requires_continuity(continuity_mode, clip) else (
            clip.reference_clip or (experiment.clips[index-1].name if index else ''))
        result[clip.name] = parent
    return result


def repair_plan(experiment, failed_clip, *, continuity_mode='independent', max_regenerations=1):
    dependencies = dependency_map(experiment, continuity_mode)
    if failed_clip not in dependencies or max_regenerations < 1:
        raise ValueError('Repair needs a known clip and a positive regeneration budget')
    invalid = {failed_clip}
    for clip, parent in dependencies.items():
        if parent in invalid:
            invalid.add(clip)
    return {'failed_clip': failed_clip, 'invalidate': [c for c in dependencies if c in invalid],
            'preserve': [c for c in dependencies if c not in invalid],
            'max_regenerations': max_regenerations,
            'note': 'Downstream evidence must be reviewed again after a changed state. No regeneration was started.'}


def choose_profile(profiles, *, duration, minimum_area, gpu_limits, max_wall_seconds=None):
    """Select only measured successful profiles; never infer a fitting duration."""
    if not math.isfinite(duration) or duration <= 0 or minimum_area <= 0 or not gpu_limits or any(not math.isfinite(v) or v <= 0 for v in gpu_limits):
        raise ValueError('Profile requirements and GPU limits must be finite and positive')
    candidates = []
    for profile in profiles:
        if profile.get('status') != 'success' or not profile.get('measurement_id'):
            continue
        try:
            seconds, area, wall = float(profile['seconds']), int(profile['area']), float(profile['wall_seconds'])
            peaks = profile['peak_allocated_bytes']
            finite = all(math.isfinite(v) and v > 0 for v in (seconds, area, wall))
            fits = len(peaks) == len(gpu_limits) and all(0 < float(p) <= limit for p, limit in zip(peaks, gpu_limits))
        except (KeyError, TypeError, ValueError):
            continue
        if finite and fits and seconds >= duration and area >= minimum_area and (max_wall_seconds is None or wall <= max_wall_seconds):
            candidates.append(profile)
    return min(candidates, key=lambda p: float(p['wall_seconds'])) if candidates else None
