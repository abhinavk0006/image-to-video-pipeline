"""Extract deterministic handoff frames from generated video clips."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path


class FrameExtractionError(RuntimeError):
	"""Raised when a frame cannot be extracted from a video file."""


@dataclass(slots=True)
class FrameExtractor:
	"""Extract frames by decoding the stream, without ffprobe or timestamp seeks.

	Decoding sequentially avoids MP4 duration/index discrepancies at the end of
	short generated clips. Only a small ring buffer is retained in memory.
	"""

	frame_offset_seconds: float = 0.50
	default_fps: float = 16.0

	def extract_last_frame(self, video_path: str | Path, output_path: str | Path) -> Path:
		"""Save the default near-end handoff frame."""
		return self.extract_frame_before_end(
			video_path, output_path, offset_seconds=self.frame_offset_seconds
		)

	def extract_frame_before_end(
		self,
		video_path: str | Path,
		output_path: str | Path,
		*,
		offset_seconds: float,
	) -> Path:
		"""Save a decoded frame the requested number of seconds before EOF.

		An offset of zero selects the final decoded frame. Positive offsets are
		rounded to the nearest source frame, clamped to the first frame for clips
		shorter than the requested offset.
		"""
		input_path = Path(video_path)
		destination_path = Path(output_path)
		if not input_path.is_file():
			raise FrameExtractionError(f"Video file does not exist: {input_path}")
		if offset_seconds < 0:
			raise FrameExtractionError("Frame offset must be non-negative")
		destination_path.parent.mkdir(parents=True, exist_ok=True)
		destination_path.unlink(missing_ok=True)

		try:
			import imageio_ffmpeg
			from PIL import Image
		except ImportError as error:
			raise FrameExtractionError(
				"Frame extraction requires imageio-ffmpeg and Pillow; install them in the pipeline environment."
			) from error

		reader = None
		try:
			reader = imageio_ffmpeg.read_frames(str(input_path), pix_fmt="rgb24")
			metadata = next(reader)
			fps = float(metadata.get("fps") or self.default_fps)
			if fps <= 0:
				fps = self.default_fps
			frames_back = max(0, int(round(offset_seconds * fps)))
			window = deque(maxlen=frames_back + 1)
			for frame in reader:
				window.append(frame)
			if not window:
				raise FrameExtractionError(f"Video contains no decodable frames: {input_path}")
			# With a full window, the oldest entry is exactly `frames_back` from EOF.
			# If the clip is shorter than the requested offset, use its first frame.
			selected = window[0]
			Image.fromarray(selected).save(destination_path)
		except FrameExtractionError:
			raise
		except Exception as error:
			destination_path.unlink(missing_ok=True)
			detail = str(error).strip() or type(error).__name__
			raise FrameExtractionError(
				f"Could not decode a frame {offset_seconds:g}s before EOF from {input_path}: {detail}"
			) from error
		finally:
			if reader is not None:
				reader.close()

		if not destination_path.is_file() or destination_path.stat().st_size == 0:
			raise FrameExtractionError(
				f"Frame extraction produced no image at {destination_path}"
			)
		return destination_path


def extract_last_frame(video_path: str | Path, output_path: str | Path) -> Path:
	"""Convenience wrapper for extracting the last frame."""
	return FrameExtractor().extract_last_frame(video_path, output_path)
