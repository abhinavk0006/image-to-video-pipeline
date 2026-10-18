"""Technical decode checks must never be mistaken for scientific approval."""
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from benchmark import collect_run, review_status, summarize
from test_frame_handoffs import write_test_video


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.videos = self.root / "videos"
        self.videos.mkdir()
        self.reports = self.root / "report"
        write_test_video(self.videos / "step.mp4")
        shutil.copy2(self.videos / "step.mp4", self.videos / "final_video.mp4")
        self.knowledge = self.root / "experiment.json"
        self.knowledge.write_text(json.dumps({"name": "test", "clips": [{"name": "step"}]}))

    def tearDown(self):
        self.temp.cleanup()

    def test_decodable_video_remains_scientifically_pending(self):
        result = collect_run(self.knowledge, self.videos, self.reports, wall_seconds=10)
        self.assertTrue(result["technical_sequence_complete"])
        self.assertEqual(result["scientific_reviews_pending"], 1)
        self.assertIsNone(result["accepted_seconds_per_wall_second"])
        self.assertEqual(result["generated_video_seconds"], 2.5)
        self.assertEqual(result["clips"][0]["handoffs"]["final"]["frame_index"], 24)

    def test_reviewed_correct_seconds_count_and_wrong_science_does_not(self):
        collect_run(self.knowledge, self.videos, self.reports)
        path = self.reports / "scientific_review.json"
        review = json.loads(path.read_text())
        checks = review["clips"]["step"]["checks"]
        checks.update(dict.fromkeys(checks, True))
        path.write_text(json.dumps(review))
        result = collect_run(self.knowledge, self.videos, self.reports, wall_seconds=10)
        self.assertEqual(result["accepted_seconds_per_wall_second"], .25)
        checks["observation_on_correct_target"] = False
        path.write_text(json.dumps(review))
        result = collect_run(self.knowledge, self.videos, self.reports, wall_seconds=10)
        self.assertEqual(result["accepted_seconds_per_wall_second"], 0)
        self.assertEqual(result["clips"][0]["scientific_status"], "failed")

    def test_wrong_duration_and_missing_dependency_fail_technical_check(self):
        self.knowledge.write_text(json.dumps({"name": "test", "clips": [
            {"name": "step", "reference_clip": "not_generated", "duration_seconds": 3}]}))
        result = collect_run(self.knowledge, self.videos, self.reports)
        self.assertFalse(result["technical_sequence_complete"])

    def test_oom_failure_is_a_reportable_result(self):
        (self.videos / "step.mp4").unlink()
        log = self.root / "generation.log"
        log.write_text('error_type="cuda_oom" CUDA out of memory')
        result = collect_run(self.knowledge, self.videos, self.reports,
            generation_status="failed", wall_seconds=10, log_path=log)
        self.assertTrue(result["cuda_oom_detected"])
        self.assertFalse(result["technical_sequence_complete"])
        self.assertEqual(result["decodable_clips"], 0)

    def test_review_requires_actual_booleans(self):
        with self.assertRaises(ValueError):
            review_status({"source_verified": "false"})

    def test_summary_refreshes_from_reviewed_case_report(self):
        case = self.root / "baseline"
        case.mkdir()
        (case / "case.json").write_text(json.dumps({"settings": {"seconds": 2, "max_area": 345600}, "exit_code": 0}))
        result = collect_run(self.knowledge, self.videos, case / "report", wall_seconds=10)
        rows = summarize(self.root)
        self.assertEqual(rows[0]["scientific_reviews_pending"], 1)
        self.assertIsNone(rows[0]["gpu0_peak_allocated_gib"])
        self.assertTrue((self.root / "summary.csv").exists())


if __name__ == "__main__":
    unittest.main()
