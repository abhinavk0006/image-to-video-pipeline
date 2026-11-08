"""CPU-only notebook readiness checks; a reference image is not a volume measurement."""
from pathlib import Path
import hashlib


def build_preflight(*, image_name, image_path=None, input_roots=('/kaggle/input', '/kaggle/working'), material_report=None, material_mode='strict'):
    if material_mode not in {'strict', 'warn'}:
        raise ValueError('material_mode must be strict or warn')
    report = {'schema_version': 1, 'ready': False, 'issues': [], 'warnings': [],
              'scientifically_verified': False, 'image': None, 'material_mode': material_mode}
    roots = [Path(root) for root in input_roots]
    if image_path:
        candidates = [Path(image_path).expanduser()]
    else:
        supported = {'.png', '.jpg', '.jpeg', '.webp'}
        # Accept a matching stem in a supported format; never pick an unrelated image.
        candidates = sorted({p.resolve() for root in roots if root.exists()
                             for p in root.rglob('*') if p.is_file()
                             and p.stem.casefold() == Path(image_name).stem.casefold()
                             and p.suffix.lower() in supported})
    if len(candidates) != 1:
        report['issues'].append(f'Expected one reference matching {image_name!r}; found {len(candidates)}. Upload it, or set INITIAL_IMAGE_PATH to its exact path. No dry-salt or unrelated-image fallback is used.')
        report['reference_candidates'] = [str(p) for p in candidates[:20]]
    else:
        path = candidates[0]
        try:
            from PIL import Image
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                width, height = image.size
            if min(width, height) < 64:
                raise ValueError('Reference must be at least 64 pixels on each axis')
            report['image'] = {'path': str(path.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                               'width': width, 'height': height, 'volume_measured': False,
                               'scientific_starting_state_reviewed': False}
            report['warnings'].append('Review reference contents against the selected starting state; file validation cannot measure mL or establish chemistry.')
        except (OSError, ValueError, Image.DecompressionBombError) as error:
            report['issues'].append(f'Invalid reference {path}: {error}')
    material = material_report or {}
    report['material_status'] = material.get('status', 'missing')
    unresolved = [{'clip': row['clip'], 'issues': row['issues']} for row in material.get('clips', []) if row.get('issues')]
    report['unresolved_material_details'] = unresolved
    if material.get('status') not in {'numerically_consistent', 'needs_source_detail'}:
        report['issues'].append('Build a valid material plan before generation.')
    elif material.get('status') == 'needs_source_detail':
        message = 'Material quantities or vessel limits remain unresolved. Supply source-backed MATERIAL_OVERRIDES; warn mode allows only an unvalidated visual trial.'
        report['issues' if material_mode == 'strict' else 'warnings'].append(message)
    report['ready'] = not report['issues']
    return report
