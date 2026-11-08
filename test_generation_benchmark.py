import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from generation_benchmark import run_benchmarked

class AutomaticBenchmarkTests(unittest.TestCase):
    def test_success_retains_manifest_without_inventing_scores(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); k=root/'experiment.json'
            k.write_text(json.dumps({'name':'x','clips':[{'name':'c'}]}))
            with patch('generation_benchmark.collect_run') as collect:
                report,_=run_benchmarked(['generator'],knowledge_file=k,experiment_dir=root/'videos',report_root=root/'reports',cwd=root,action=lambda:None)
            data=json.loads((report/'run.json').read_text())
            self.assertEqual(data['generation_status'],'success')
            self.assertIsNone(data['scientific_accuracy'])
            self.assertEqual(data['visual_metric_status'],'no_generated_videos')
            self.assertEqual(len(data['missing_videos']),1)
            collect.assert_called_once()

    def test_failure_is_preserved_and_still_collected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); k=root/'experiment.json'
            k.write_text(json.dumps({'name':'x','clips':[{'name':'c'}]}))
            def fail():
                raise ValueError('failed generation')
            with patch('generation_benchmark.collect_run') as collect:
                with self.assertRaisesRegex(ValueError,'failed generation'):
                    run_benchmarked(['generator'],knowledge_file=k,experiment_dir=root/'videos',report_root=root/'reports',cwd=root,action=fail)
            data=json.loads(next((root/'reports').glob('*/run.json')).read_text())
            self.assertEqual(data['generation_status'],'failed')
            collect.assert_called_once()

    def test_real_video_collection_metric_preparation_and_resume_label(self):
        from test_frame_handoffs import write_test_video
        import shutil
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); videos=root/'videos'; videos.mkdir()
            k=root/'experiment.json'
            k.write_text(json.dumps({'name':'x','clips':[{'name':'c','duration_seconds':2}]}))
            def generate():
                write_test_video(videos/'c.mp4',count=33,fps=16)
                shutil.copyfile(videos/'c.mp4',videos/'final_video.mp4')
            cwd=Path(__file__).resolve().parent
            with patch('generation_benchmark.shutil.which',return_value=None):
                report,_=run_benchmarked(['integration'],knowledge_file=k,experiment_dir=videos,report_root=root/'reports',cwd=cwd,action=generate)
                resumed,_=run_benchmarked(['integration'],knowledge_file=k,experiment_dir=videos,report_root=root/'reports',cwd=cwd,action=lambda:None)
            data=json.loads((report/'run.json').read_text())
            metrics=json.loads((report/'vbench/metric_run.json').read_text())
            self.assertEqual(data['visual_metric_status'],'prepared_not_scored')
            self.assertEqual(metrics['status'],'prepared')
            self.assertEqual(len(metrics['commands']),6)
            self.assertTrue((report/'benchmark.json').is_file())
            self.assertTrue((report/'scientific_review.json').is_file())
            self.assertTrue(data['eligible_for_cold_generation_timing'])
            self.assertFalse(json.loads((resumed/'run.json').read_text())['eligible_for_cold_generation_timing'])
