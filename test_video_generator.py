"""Focused tests for image-to-video generation fallback behavior."""

import subprocess
import tempfile
import unittest
import json
from pathlib import Path
from unittest import mock

from kaggle_wan_wrapper import configured_areas
from models import PromptBundle
from video_generator import (
	LocalVideoGenerator,
	VideoGenerationError,
	VideoGenerationRequest,
)


class VideoGeneratorFallbackTests(unittest.TestCase):
	def setUp(self) -> None:
		self.temp_dir = tempfile.TemporaryDirectory()
		self.root = Path(self.temp_dir.name)
		self.image = self.root / "input.png"
		self.image.write_bytes(b"image")
		self.output = self.root / "clip.mp4"
		self.request = VideoGenerationRequest(
			input_image_path=self.image,
			output_video_path=self.output,
			prompt_bundle=PromptBundle(motion_prompt="move"),
			clip_name="clip-1",
		)

	def tearDown(self) -> None:
		self.temp_dir.cleanup()

	def test_subprocess_retries_cuda_oom_at_lower_area(self) -> None:
		generator = LocalVideoGenerator(
			script_path="generate.py",
			max_area=399360,
			fallback_areas=(200704,),
		)
		generator._use_daemon = False

		def run(command, **kwargs):
			if command[-1] == "399360":
				raise subprocess.CalledProcessError(
					1, command, stderr="torch.cuda.OutOfMemoryError: CUDA out of memory"
				)
			self.output.write_bytes(b"video")

		with mock.patch("video_generator.subprocess.run", side_effect=run) as mocked:
			result = generator.generate_clip_video(self.request)

		self.assertEqual(result, self.output)
		self.assertEqual(
			[mock_call.args[0][-1] for mock_call in mocked.call_args_list],
			["399360", "200704"],
		)

	def test_non_oom_failure_is_not_retried(self) -> None:
		generator = LocalVideoGenerator(
			script_path="generate.py",
			max_area=399360,
			fallback_areas=(200704,),
		)
		generator._use_daemon = False
		error = subprocess.CalledProcessError(1, ["generate.py"], stderr="invalid prompt")

		with mock.patch("video_generator.subprocess.run", side_effect=error) as mocked:
			with self.assertRaisesRegex(VideoGenerationError, "invalid prompt"):
				generator.generate_clip_video(self.request)

		mocked.assert_called_once()

	def test_daemon_retries_without_killing_process(self) -> None:
		class FakeStdin:
			def __init__(self):
				self.writes = 0

			def write(self, value):
				self.writes += 1
				if self.writes == 2:
					self.output.write_bytes(b"video")
				return len(value)

			def flush(self):
				pass

		class FakeStdout:
			def __init__(self):
				self.lines = iter([
					"READY\n",
					json.dumps({"status": "error", "error": "CUDA out of memory"}) + "\n",
					'{"status":"success"}\n',
				])

			def readline(self):
				return next(self.lines)

		class FakeProcess:
			def __init__(self, output):
				self.stdin = FakeStdin()
				self.stdin.output = output
				self.stdout = FakeStdout()
				self.killed = False

			def kill(self):
				self.killed = True

		process = FakeProcess(self.output)
		generator = LocalVideoGenerator(
			script_path="generate.py",
			max_area=399360,
			fallback_areas=(200704,),
		)
		with mock.patch("video_generator.subprocess.Popen", return_value=process):
			self.assertEqual(generator.generate_clip_video(self.request), self.output)
		self.assertFalse(process.killed)
		self.assertEqual(process.stdin.writes, 2)

	def test_areas_must_decrease(self) -> None:
		generator = LocalVideoGenerator(max_area=100, fallback_areas=(100,))
		with self.assertRaisesRegex(VideoGenerationError, "unique"):
			generator._generation_areas()

	def test_wrapper_area_configuration_is_ordered(self) -> None:
		args = type("Args", (), {"max_area": 399360, "max_area_fallback": [200704]})()
		self.assertEqual(configured_areas(args), (399360, 200704))


if __name__ == "__main__":
	unittest.main()
