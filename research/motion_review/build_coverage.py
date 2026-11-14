"""Suggest source-step to motion-family matches; never assert scientific approval."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parent
DATA=ROOT.parents[2]/'Edu-video-gen-dataset/data/chemistry_experiments.json'
PATTERNS={'dropper':r'dropwise|drops|\badd\b|acidify|alkaline|neutralisation','pour':r'pour|transfer|combine|stock|place .*ml|take .*ml|fill|rinse','stir':r'stir','pipette_aspiration':r'pipette','solid_transfer':r'weigh|crystals|powder|solid','paper_contact':r'litmus|paper.*drop','gas_test_position':r'gas|mouth|fumes','tube_shaking':r'shake|shaking','swirl_flask':r'swirl','filter_washing':r'filter|wash','sample_withdrawal':r'aliquot|portion','thermometer_position':r'temperature|thermometer','bath_lowering':r'water bath|sand bath','paper_dipping':r'chromatograph|mobile phase','electrode_placement':r'half-cell|electrode|set up','salt_bridge_placement':r'salt bridge','burette_delivery':r'titrat|burette','membrane_suspension':r'cone|membrane|cellophane','paper_contact':r'litmus|paper.*drop|apply|spot','stopper_placement':r'stopper|condenser','apparatus_wait':r'cool|stand|minutes|heat|boil|dry|dialysis|record|measure|equilibrate|develop|dissolv'}
def main():
 data=json.loads(DATA.read_text(encoding='utf-8'));rows=[]
 for record in data['experiments']:
  for variant in record.get('protocol_variants',[]):
   for step in variant['procedure_steps']:
    text=step['instruction'].lower();matches=[name for name,pattern in PATTERNS.items() if re.search(pattern,text)]
    rows.append({'experiment_id':record['experiment_id'],'protocol':variant['variant_id'],'step_id':step['step_id'],'source_step_sha256':hashlib.sha256(json.dumps(step,sort_keys=True,ensure_ascii=False).encode()).hexdigest(),'candidate_motions':matches,'requires_new_template':not bool(matches) or ('prepare' in text and ('cellophane' in text or 'parchment' in text)),'requires_apparatus_fit':True,'requires_topology_validation':any(word in text for word in ('half-cell','electrode','salt bridge')),'status':'suggested_requires_apparatus_fit_and_review'})
 result={'status':'draft_coverage_not_validated_mapping','source_steps':rows,'physics_source_pending':True}
 (ROOT/'source_motion_candidates.json').write_text(json.dumps(result,indent=2)+'\n')
 print('Mapped candidates for',len(rows),'source steps;',sum(r['requires_new_template'] for r in rows),'need new templates; all require apparatus fitting and review')
if __name__=='__main__':main()
