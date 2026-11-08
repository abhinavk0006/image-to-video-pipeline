"""Load draft image/procedure pilots with exact source and asset identity checks."""
import copy
import hashlib
import json
from pathlib import Path
from source_quantities import bind_source_quantities


def load_reference_pilot(dataset_root, pilot_id, *, duration_seconds=2.0):
    root=Path(dataset_root).resolve()
    registry=json.loads((root/'assets/reference_pilots/manifest.json').read_text(encoding='utf-8'))
    pilot=registry['pilots'][pilot_id]
    records=json.loads((root/'data/chemistry_experiments.json').read_text(encoding='utf-8'))['experiments']
    record=next(r for r in records if r['experiment_id']==pilot['experiment_id'])
    variant=next(v for v in record['protocol_variants'] if v['variant_id']==pilot['protocol_variant'])
    steps={s['step_id']:s for s in variant['procedure_steps']}
    for sid,expected in pilot['source_step_hashes'].items():
        actual=hashlib.sha256(json.dumps(steps[int(sid)],sort_keys=True,ensure_ascii=False).encode()).hexdigest()
        if actual!=expected:
            raise ValueError(f'{pilot_id}: source step {sid} changed; review and rebind the pilot before generation')
    image=(root/pilot['image']['path']).resolve()
    if not image.is_relative_to(root) or hashlib.sha256(image.read_bytes()).hexdigest()!=pilot['image']['sha256']:
        raise ValueError(f'{pilot_id}: reference asset is outside dataset or its hash changed')
    clips=[]
    for index,spec in enumerate(pilot['clips']):
        source=steps[spec['source_step_id']]
        action=spec['action']
        if spec['name']=='swirl':
            action='Gently swirl the existing beaker once while keeping all liquid inside.'
        clips.append({'name':spec['name'],'scene':{'camera':'Fixed scientific close view matching the supplied image'},
            'action':action,'state_before':spec['state_before'],'state_after':spec['state_after'],
            'duration_seconds':duration_seconds,'state_track':pilot_id+'_sample',
            'input_frame_path':str(image) if index==0 else None,'reference_clip':clips[-1]['name'] if index else '',
            'reference_mode':'chained' if index else 'independent','continuity_required':index>0,
            'handoff_policy':'final' if index else 'none','liquid_state':copy.deepcopy(spec['liquid_state']),
            'tool_contact_policy':spec.get('tool_contact_policy', 'above_rim' if pilot_id=='chloride_confirmation' or spec['name']=='acid_delivery' else 'source_defined'),
            'target_object':spec.get('target_object', 'The existing sample and apparatus in the reference image'),
            'preserve_objects':spec.get('preserve_objects', ['apparatus count, geometry and position','background and any observation marker']),
            'action_pacing':'One deliberate action only; preserve laboratory waiting times as process conditions, not generated duration.',
            'metadata':{'experiment_id':record['experiment_id'],'protocol_variant':pilot['protocol_variant'],
                'source_ids':[e['source_id'] for e in source['source_evidence']],
                'source_evidence':source['source_evidence'],'source_step_snapshot':source,'pilot_scope':pilot['scope'],
                'manual_review_required':True,'requires_clip_split':False,
                'atomic_rendering_adaptation':True,
                'quantities':[q for q in source['quantities'] if 'quantity_reagents' not in spec or q['reagent'] in spec['quantity_reagents']] if spec['apply_source_quantities'] else [],
                'science_spec':{'source_reviewed':False,'entities':[], 'preconditions':[spec['state_before']],
                    'requirements':[{'id':'action','description':action,'critical':True,'evidence_kind':'interval'},
                                    {'id':'observation','description':spec['state_after'],'critical':True,'evidence_kind':'interval'}],
                    'temporal_order':[]}}})
    plan={'name':pilot_id,'subject':record['subject'],'educational_goal':pilot['scope'],
          'description':pilot['starting_state'],'conclusion_text':'Illustrative procedure excerpt. Scientific and temporal correctness require independent source-linked review.',
          'metadata':{'reference_pilot':{'id':pilot_id,'image_sha256':pilot['image']['sha256'],'source_step_hashes':pilot['source_step_hashes'],
                        'review_status':'draft','scope':pilot['scope']},'scientifically_verified':False},'clips':clips}
    return bind_source_quantities(plan)
