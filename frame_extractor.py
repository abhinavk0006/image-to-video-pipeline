"""Extract the final frame from a generated video clip.

This module stays intentionally small: it only reads a video file and writes
the last frame to disk so the next clip can reuse it as input.
"""

from __future__ import annotations

import subprocess
try:
	from imageio_ffmpeg import get_ffmpeg_exe
except Exception:
	get_ffmpeg_exe = None
from dataclasses import dataclass
from pathlib import Path


class FrameExtractionError(RuntimeError):
	"""Raised when a frame cannot be extracted from a video file."""


@dataclass(slots=True)
class FrameExtractor:
	"""Extracts the final frame from a clip using local ffmpeg tooling."""

	# small offset from end; larger than a few milliseconds to reliably hit a frame
	frame_offset_seconds: float = 0.50

	def extract_last_frame(self, video_path: str | Path, output_path: str | Path) -> Path:
		"""Save a stable near-end handoff frame to `output_path`."""
		return self.extract_frame_before_end(
			video_path,
			output_path,
			offset_seconds=self.frame_offset_seconds,
		)

	def extract_frame_before_end(
		self,
		video_path: str | Path,
		output_path: str | Path,
		*,
		offset_seconds: float,
	) -> Path:
		"""Save the frame `offset_seconds` before the end of a video."""

		input_path = Path(video_path)
		destination_path = Path(output_path)

		if not input_path.exists():
			raise FrameExtractionError(f"Video file does not exist: {input_path}")
		if offset_seconds < 0:
			raise FrameExtractionError("Frame offset must be non-negative")

		destination_path.parent.mkdir(parents=True, exist_ok=True)

		frame_count = self._get_frame_count(input_path)
		fps = self._get_frame_rate(input_path)
		frames_back = max(1, int(round(offset_seconds * fps)))
		target_index = max(0, frame_count - frames_back - 1)

		ffmpeg_exe = get_ffmpeg_exe() if get_ffmpeg_exe is not None else "ffmpeg"
		# Select a decoded frame by index. This avoids unreliable MP4 duration
		# metadata and timestamp seeks that can land outside the frame range.
		command = [
			ffmpeg_exe,
			"-y",
			"-i",
			str(input_path),
			"-vf",
			f"select=eq(n\\,{target_index})",
			"-frames:v",
			"1",
			"-vsync",
			"0",
			"-f",
			"image2",
			str(destination_path),
		]

		try:
			result = subprocess.run(
				command,
				check=True,
				capture_output=True,
				text=True,
				errors="replace",
			)
		except FileNotFoundError as error:
			ffmpeg_exe = get_ffmpeg_exe() if get_ffmpeg_exe is not None else "ffmpeg"
			raise FrameExtractionError(f"{ffmpeg_exe} is not available on this system") from error
		except subprocess.CalledProcessError as error:
			raise FrameExtractionError(
				f"Failed to extract last frame from {input_path}: {error.stderr.strip()}"
			) from error

		if not destination_path.exists():
			stderr = result.stderr.strip() if "result" in locals() else ""
			raise FrameExtractionError(
				f"Frame extraction completed but no file was written: "
				f"{destination_path}. ffmpeg output: {stderr[-1000:]}"
			)

		return destination_path

	def _get_frame_count(self, video_path: Path) -> int:
		"""Return the number of decodable video frames."""
		command = [
			"ffprobe",
			"-v",
			"error",
			"-count_frames",
			"-select_streams",
			"v:0",
			"-show_entries",
			"stream=nb_read_frames",
			"-of",
			"default=noprint_wrappers=1:nokey=1",
			str(video_path),
		]
		try:
			result = subprocess.run(command, check=True, capture_output=True, text=True, errors="replace")
			count = int(result.stdout.strip())
		except (FileNotFoundError, subprocess.CalledProcessError, ValueError) as error:
			raise FrameExtractionError(f"Could not count video frames for {video_path}") from error
		if count < 1:
			raise FrameExtractionError(f"Video contains no decodable frames: {video_path}")
		return count

	def _get_frame_rate(self, video_path: Path) -> float:
		"""Return the source frame rate, falling back to 16 FPS."""
		command = [
			"ffprobe",
			"-v",
			"error",
			"-select_streams",
			"v:0",
			"-show_entries",
			"stream=avg_frame_rate",
			"-of",
			"default=noprint_wrappers=1:nokey=1",
			str(video_path),
		]
		try:
			result = subprocess.run(command, check=True, capture_output=True, text=True, errors="replace")
			numerator, denominator = result.stdout.strip().split("/", 1)
			rate = float(numerator) / float(denominator)
			if rate > 0:
				return rate
		except (FileNotFoundError, subprocess.CalledProcessError, ValueError, ZeroDivisionError):
			pass
		return 16.0

	def _get_video_duration(self, video_path: Path) -> float:
		# Prefer ffprobe when available, otherwise fall back to ffmpeg's stderr parsing
		ffprobe_cmd = [
			"ffprobe",
			"-v",
			"error",
			"-show_entries",
			"format=duration",
			"-of",
			"default=noprint_wrappers=1:nokey=1",
			str(video_path),
		]

		# Try ffprobe first
		try:
			result = subprocess.run(ffprobe_cmd, check=True, capture_output=True, text=True, errors="replace")
			duration_text = result.stdout.strip()
			if duration_text:
				return float(duration_text)
		except subprocess.CalledProcessError as error:
			raise FrameExtractionError(
				f"Failed to read duration for {video_path}: {error.stderr.strip()}"
			) from error
		except FileNotFoundError:
			# ffprobe missing; try ffmpeg and parse stderr
			ffmpeg_exe = get_ffmpeg_exe() if get_ffmpeg_exe is not None else "ffmpeg"
			cmd = [ffmpeg_exe, "-i", str(video_path)]
			try:
				proc = subprocess.run(cmd, check=False, capture_output=True, text=True, errors="replace")
			except FileNotFoundError as e:
				raise FrameExtractionError("ffmpeg is not available on this system") from e
			err = proc.stderr or proc.stdout
			# look for Duration: HH:MM:SS.xx
			import re
			m = re.search(r"Duration: (\d+):(\d+):(\d+\.\d+)", err)
			if not m:
				raise FrameExtractionError(f"Could not determine duration for {video_path}")
			h, mi, s = m.groups()
			return float(h) * 3600.0 + float(mi) * 60.0 + float(s)


def extract_last_frame(video_path: str | Path, output_path: str | Path) -> Path:
	"""Convenience wrapper for extracting the last frame."""

	return FrameExtractor().extract_last_frame(video_path, output_path)
