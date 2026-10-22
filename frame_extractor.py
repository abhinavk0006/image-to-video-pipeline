"""Extract deterministic handoff frames from generated video clips."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from video_decode import read_rgb_frames
import json
import math


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

	def extract_handoff_frame(self, video_path, output_path, *, mode="near_end", offset_seconds=None):
		"""Select by decoded frame count, never by the requested generation duration."""
		if mode not in {"near_end", "final", "mid", "offset"}:
			raise FrameExtractionError(f"Unknown handoff mode: {mode}")
		if mode == "mid":
			return self.extract_frame_before_end(video_path, output_path, offset_seconds=0, midpoint=True, selection_mode=mode)
		if mode == "offset" and offset_seconds is None:
			raise FrameExtractionError("offset mode requires offset_seconds")
		return self.extract_frame_before_end(
			video_path, output_path,
			selection_mode=mode,
			offset_seconds=0 if mode == "final" else (
				self.frame_offset_seconds if mode == "near_end" else offset_seconds
			),
		)

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
		midpoint: bool = False,
		selection_mode: str | None = None,
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
		if not math.isfinite(offset_seconds) or offset_seconds < 0:
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
			reader = read_rgb_frames(input_path)
			metadata = next(reader)
			fps = float(metadata.get("fps") or self.default_fps)
			if not math.isfinite(fps) or fps <= 0:
				fps = self.default_fps
			if midpoint:
				# A second sequential pass keeps memory bounded, including for long clips.
				frame_count = sum(1 for _ in reader)
				reader.close()
				reader = read_rgb_frames(input_path)
				metadata = next(reader)
				selected_index = (frame_count - 1) // 2
				selected = None
				for index, frame in enumerate(reader):
					if index == selected_index:
						selected = frame
						break
			else:
				frames_back = max(0, int(round(offset_seconds * fps)))
				window = deque(maxlen=frames_back + 1)
				frame_count = 0
				for frame in reader:
					window.append(frame)
					frame_count += 1
				selected = window[0] if window else None
				selected_index = max(0, frame_count - 1 - frames_back)
			if selected is None:
				raise FrameExtractionError(f"Video contains no decodable frames: {input_path}")
			# imageio-ffmpeg yields packed RGB bytes, not NumPy arrays.
			Image.frombytes("RGB", tuple(metadata["size"]), selected).save(destination_path)
			destination_path.with_suffix(destination_path.suffix + ".json").write_text(
				json.dumps({"source": str(input_path), "frame_index": selected_index,
					"decoded_frames": frame_count, "fps": fps,
					"timestamp_seconds": selected_index / fps,
					"mode": selection_mode or ("mid" if midpoint else "offset"), "offset_seconds": offset_seconds}, indent=2),
				encoding="utf-8",
			)
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
