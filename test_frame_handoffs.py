"""Regression checks using real RGB -> H.264 -> decoded handoff images."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import imageio_ffmpeg
from PIL import Image

from config import PipelineConfig
from frame_extractor import FrameExtractor, FrameExtractionError
from knowledge_loader import load_experiment_from_json
from models import Clip, Experiment, PromptBundle, Scene
from pipeline import ExperimentPipeline, clip_requires_continuity
from prompt_builder import PromptBuilder
from stitcher import VideoStitcher


def write_test_video(path, count=25, fps=10, size=(32, 32)):
    writer = imageio_ffmpeg.write_frames(str(path), size, fps=fps,
        codec="libx264", pix_fmt_out="yuv444p", macro_block_size=1, ffmpeg_log_level="error")
    writer.send(None)
    for index in range(count):
        writer.send(Image.new("RGB", size, (index * 8,) * 3).tobytes())
    writer.close()


class HandoffTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.video = self.root / "source.mp4"
        write_test_video(self.video)

    def tearDown(self):
        self.temp.cleanup()

    def test_real_final_near_end_mid_and_explicit_offset(self):
        for mode, offset, expected in [("final", None, 24), ("near_end", None, 19),
                                       ("mid", None, 12), ("offset", .3, 21), ("offset", 100, 0)]:
            with self.subTest(mode=mode, offset=offset):
                output = self.root / f"{mode}-{offset}.png"
                FrameExtractor().extract_handoff_frame(self.video, output, mode=mode, offset_seconds=offset)
                metadata = json.loads(output.with_suffix(".png.json").read_text())
                self.assertEqual(metadata["frame_index"], expected)
                self.assertEqual(metadata["decoded_frames"], 25)
                self.assertEqual(metadata["mode"], mode)
                with Image.open(output) as image:
                    self.assertLessEqual(abs(image.getpixel((10, 10))[0] - expected * 8), 3)

    def test_empty_and_invalid_requests_fail_explicitly(self):
        with self.assertRaises(FrameExtractionError):
            FrameExtractor().extract_handoff_frame(self.video, self.root / "x.png", mode="offset")
        with self.assertRaises(FrameExtractionError):
            FrameExtractor().extract_frame_before_end(self.video, self.root / "x.png", offset_seconds=float("nan"))
        broken = self.root / "broken.mp4"
        broken.write_bytes(b"not an mp4")
        with self.assertRaises(FrameExtractionError):
            FrameExtractor().extract_handoff_frame(broken, self.root / "x.png", mode="mid")

    def test_consumers_choose_distinct_frames_from_same_producer_and_resume(self):
        image = self.root / "input.png"
        Image.new("RGB", (32, 32)).save(image)
        clips = [Clip(name="source", image_state=Scene("", "", ""),
                      motion_prompt=PromptBundle(motion_prompt="one action", clip_duration_seconds=99),
                      input_frame_path=str(image))]
        for mode in ("mid", "final"):
            clips.append(Clip(name=mode, image_state=Scene("", "", ""),
                motion_prompt=PromptBundle(motion_prompt="hold"), reference_clip="source",
                reference_mode="chained", continuity_required=True, handoff_policy=mode))
        generator = Mock(spec=["generate_clip_video", "stop_daemon"])
        requests = []
        def generate(request):
            requests.append(request)
            write_test_video(Path(request.output_video_path))
            return Path(request.output_video_path)
        generator.generate_clip_video.side_effect = generate
        stitcher = Mock()
        stitcher.stitch_clips.return_value = self.root / "final.mp4"
        pipeline = ExperimentPipeline(config=PipelineConfig(base_output_dir=str(self.root),
            experiment_name="run", continuity_mode="chain"), video_generator=generator, stitcher=stitcher)
        experiment = Experiment(name="test", clips=clips)
        pipeline.run(experiment)
        for request, expected in zip(requests[1:], (12, 24)):
            meta = json.loads(Path(str(request.input_image_path) + ".json").read_text())
            self.assertEqual(meta["frame_index"], expected)
        pipeline.run(experiment)
        self.assertEqual(len(requests), 3, "Unchanged inputs should resume without generation")
        clips[1].motion_prompt.motion_prompt = "different action"
        pipeline.run(experiment)
        self.assertEqual(len(requests), 4, "Changed prompt must invalidate its old output")
        self.assertTrue(list((self.root / "run").glob("mid.previous-*.mp4")))

    def test_litmus_grounding_precedes_action(self):
        clip = Clip(name="paper", image_state=Scene("", "", ""),
            motion_prompt=PromptBundle(motion_prompt="Hold damp red litmus at the test tube opening"),
            observable_target="paper turns blue; solution remains colourless")
        prompt = PromptBuilder().build_motion_prompt(Experiment(name="gas test"), clip)
        self.assertIn("paper only", prompt)
        self.assertIn("outside the liquid", prompt)
        self.assertLess(prompt.index("paper only"), prompt.index("Hold damp red litmus"))

    def test_explicit_state_dependency_overrides_global_independent_mode(self):
        clip = Clip(name="paper", image_state=Scene("", "", ""), motion_prompt=PromptBundle(),
            reference_clip="source", reference_mode="chained", continuity_required=True)
        self.assertTrue(clip_requires_continuity("independent", clip))
        clip.continuity_required = False
        # A stale boolean cannot bypass an explicitly declared generated-state reference.
        self.assertTrue(clip_requires_continuity("chain", clip))
        clip.reference_clip = ""
        clip.reference_mode = "independent"
        self.assertFalse(clip_requires_continuity("chain", clip))

    def test_mixed_resolution_fallback_clips_are_normalized_before_concat(self):
        other = self.root / "other.mp4"
        write_test_video(other, fps=12, size=(48, 32))
        output = VideoStitcher().stitch_clips([self.video, other], self.root / "joined.mp4")
        reader = imageio_ffmpeg.read_frames(str(output))
        metadata = next(reader)
        self.assertEqual(metadata["size"], (32, 32))
        self.assertEqual(metadata["fps"], 10)
        count = sum(1 for _ in reader)
        reader.close()
        self.assertGreaterEqual(count, 44)
        self.assertLessEqual(count, 48)


if __name__ == "__main__":
    unittest.main()
