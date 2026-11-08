import tempfile
import unittest
from pathlib import Path
from PIL import Image
from run_preflight import build_preflight

class PreflightTests(unittest.TestCase):
    def check(self, root, **kw):
        return build_preflight(image_name='sample.png', input_roots=[root], material_report={'status':'numerically_consistent','clips':[]}, **kw)

    def test_missing_image_is_actionable(self):
        with tempfile.TemporaryDirectory() as d:
            result=self.check(d)
            self.assertFalse(result['ready'])
            self.assertIn('INITIAL_IMAGE_PATH',result['issues'][0])

    def test_matching_jpeg_and_image_hash(self):
        with tempfile.TemporaryDirectory() as d:
            Image.new('RGB',(128,72)).save(Path(d)/'sample.jpg')
            result=self.check(d)
            self.assertTrue(result['ready'])
            self.assertEqual(len(result['image']['sha256']),64)
            self.assertFalse(result['image']['volume_measured'])

    def test_ambiguous_image_needs_explicit_selection(self):
        with tempfile.TemporaryDirectory() as d:
            for ext in ('png','jpg'):
                Image.new('RGB',(128,72)).save(Path(d)/f'sample.{ext}')
            self.assertFalse(self.check(d)['ready'])
            self.assertTrue(self.check(d,image_path=Path(d)/'sample.png')['ready'])

    def test_bad_reference_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'sample.png'
            p.write_text('not an image')
            self.assertFalse(self.check(d)['ready'])
            Image.new('RGB',(2,2)).save(p)
            self.assertFalse(self.check(d)['ready'])

    def test_combined_failures_and_warn_does_not_bypass_missing_image(self):
        with tempfile.TemporaryDirectory() as d:
            kw={'image_name':'sample.png','input_roots':[d], 'material_report':{'status':'needs_source_detail','clips':[{'clip':'c','issues':['Missing dose']} ]}}
            self.assertEqual(len(build_preflight(**kw)['issues']),2)
            self.assertFalse(build_preflight(**kw,material_mode='warn')['ready'])
            Image.new('RGB',(128,72)).save(Path(d)/'sample.png')
            result=build_preflight(**kw,material_mode='warn')
            self.assertTrue(result['ready'])
            self.assertFalse(result['scientifically_verified'])
            self.assertEqual(result['unresolved_material_details'][0]['clip'],'c')
