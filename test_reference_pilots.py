import json
from pathlib import Path
import tempfile
import unittest
from reference_pilots import load_reference_pilot
from knowledge_loader import _parse_clip
from models import Experiment
from material_state import prepare_state_tracks
from material_plan import compile_material_plan

ROOT=Path(__file__).resolve().parent.parent/'Edu-video-gen-dataset'
class PilotTests(unittest.TestCase):
    def test_all_pilots_load_and_preserve_chain(self):
        registry=json.loads((ROOT/'assets/reference_pilots/manifest.json').read_text())
        for pid,pilot in registry['pilots'].items():
            count=len(pilot['clips'])
            plan=load_reference_pilot(ROOT,pid)
            self.assertEqual(len(plan['clips']),count)
            self.assertTrue(Path(plan['clips'][0]['input_frame_path']).is_file())
            exp=Experiment(name=pid,clips=[_parse_clip(c,i) for i,c in enumerate(plan['clips'])])
            prepare_state_tracks(exp)
            report=compile_material_plan(exp)
            self.assertFalse(plan['metadata']['scientifically_verified'])
            self.assertFalse(report['scientifically_verified'])
            for i in range(1,count):
                self.assertEqual(exp.clips[i].reference_clip,exp.clips[i-1].name)
                self.assertEqual(exp.clips[i].handoff_policy,'final')
            if pid=='thiosulphate_cross':
                self.assertEqual(report['clips'][0]['expected_added_ml'],1)
                self.assertEqual(report['clips'][1]['expected_added_ml'],0)
                self.assertEqual(report['clips'][2]['expected_after_ml'],51)
            if pid=='chloride_confirmation':
                self.assertEqual(report['status'],'needs_source_detail')

    def test_changed_source_or_image_is_rejected(self):
        import shutil
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            (root/'data').mkdir()
            (root/'assets/reference_pilots').mkdir(parents=True)
            shutil.copyfile(ROOT/'data/chemistry_experiments.json',root/'data/chemistry_experiments.json')
            manifest=json.loads((ROOT/'assets/reference_pilots/manifest.json').read_text())
            pilot=manifest['pilots']['chloride_confirmation']
            original_hashes=dict(pilot['source_step_hashes'])
            pilot['source_step_hashes']['1']='changed'
            (root/'assets/reference_pilots/manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError,'source step'):
                load_reference_pilot(root,'chloride_confirmation')
            pilot['source_step_hashes']={}
            (root/'assets/reference_pilots/manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError,'source step set'):
                load_reference_pilot(root,'chloride_confirmation')
            pilot['source_step_hashes']=original_hashes
            (root/'assets/reference_pilots/manifest.json').write_text(json.dumps(manifest))
            image=root/pilot['image']['path'];image.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'hash changed'):
                load_reference_pilot(root,'chloride_confirmation')

    def test_atomic_reagents_and_reused_references_keep_separate_samples(self):
        iodine=load_reference_pilot(ROOT,'starch_iodine')
        self.assertEqual([q['reagent'] for q in iodine['clips'][0]['metadata']['quantities']],['iodine'])
        self.assertEqual(len(iodine['clips'][0]['metadata']['source_step_snapshot']['quantities']),5)
        for dose in (1,2,4):
            pid='thiosulphate_cross' if dose==1 else f'thiosulphate_cross_{dose}ml'
            plan=load_reference_pilot(ROOT,pid)
            exp=Experiment(name=pid,clips=[_parse_clip(c,i) for i,c in enumerate(plan['clips'])])
            prepare_state_tracks(exp)
            rows=compile_material_plan(exp)['clips']
            self.assertEqual(rows[0]['expected_added_ml'],dose)
            self.assertEqual(rows[-1]['expected_after_ml'],50+dose)
            self.assertEqual(rows[1]['expected_added_ml'],0)
        a=load_reference_pilot(ROOT,'iron_thiocyanate')['clips'][0]
        b=load_reference_pilot(ROOT,'iron_ferrocyanide')['clips'][0]
        self.assertEqual(a['input_frame_path'],b['input_frame_path'])
        self.assertNotEqual(a['state_track'],b['state_track'])
        self.assertEqual(a['reference_clip'],b['reference_clip'])
        self.assertEqual(a['reference_clip'],'')
        self.assertNotEqual(a['metadata']['quantities'],b['metadata']['quantities'])
        wash=load_reference_pilot(ROOT,'alum_crystal_washing')['clips'][0]
        self.assertIn('filter paper',wash['target_object'])
        litmus=load_reference_pilot(ROOT,'carboxyl_litmus')['clips'][0]
        self.assertIn('litmus paper',litmus['target_object'])
        self.assertEqual(litmus['liquid_state']['dose']['count'],1)
