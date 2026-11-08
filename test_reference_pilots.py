import copy
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
        for pid,count in [('chloride_confirmation',3),('thiosulphate_cross',3),('starch_heating_excerpt',1)]:
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
            pilot['source_step_hashes']['1']='changed'
            (root/'assets/reference_pilots/manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError,'source step'):
                load_reference_pilot(root,'chloride_confirmation')
            pilot['source_step_hashes']={}
            (root/'assets/reference_pilots/manifest.json').write_text(json.dumps(manifest))
            image=root/pilot['image']['path'];image.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'hash changed'):
                load_reference_pilot(root,'chloride_confirmation')
