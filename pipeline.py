"""Pipeline orchestration for clip-by-clip educational video generation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import hashlib
import json
import math
import time

from config import PipelineConfig
import sys
from frame_extractor import FrameExtractor
from models import Clip, Experiment
from material_state import prepare_state_tracks
from material_plan import compile_material_plan
from material_review import material_review_status
from prompt_builder import PromptBuilder
from stitcher import VideoStitcher
from video_generator import LocalVideoGenerator, VideoGenerationRequest


class PipelineError(RuntimeError):
	"""Raised when the pipeline cannot complete."""


class ManualImageRequiredError(PipelineError):
	"""Raised when a manual image is required before generation can continue."""

	def __init__(
		self,
		clip_name: str,
		expected_image_path: Path,
		image_prompt_path: Path,
		message: str,
	) -> None:
		super().__init__(message)
		self.clip_name = clip_name
		self.expected_image_path = expected_image_path
		self.image_prompt_path = image_prompt_path


def resolve_handoff_offset_seconds(
	mode: str,
	*,
	explicit_offset_seconds: float | None,
	default_offset_seconds: float,
	clip_duration_seconds: float | None = None,
) -> float:
	"""Resolve the handoff frame timing from a clip's declared state policy."""
	if mode == "near_end":
		offset = default_offset_seconds
	elif mode == "final":
		offset = 0.0
	elif mode == "mid":
		if clip_duration_seconds is None or clip_duration_seconds <= 0:
			raise PipelineError("handoff_mode='mid' requires a positive clip duration")
		offset = clip_duration_seconds / 2
	elif mode == "offset":
		if explicit_offset_seconds is None:
			raise PipelineError(
				"handoff_mode='offset' requires handoff_offset_seconds"
			)
		offset = explicit_offset_seconds
	else:
		raise PipelineError(
			"handoff_mode must be 'near_end', 'mid', 'final', or 'offset'"
		)
	if offset < 0:
		raise PipelineError("handoff_offset_seconds must be non-negative")
	return offset


def clip_requires_continuity(config_mode: str, clip: Clip) -> bool:
	"""Apply explicit per-step continuity metadata over the global mode."""
	if clip.starts_new_sample:
		return False
	if clip.reference_clip or clip.reference_mode == "chained":
		return True
	if clip.continuity_required is not None:
		return clip.continuity_required
	return config_mode == "chain" and clip.reference_policy not in {"independent", "state_keyframe"}


@dataclass(slots=True)
class ExperimentPipeline:
	"""Coordinates clip-by-clip generation from structured experiment data."""

	config: PipelineConfig = field(default_factory=PipelineConfig)
	prompt_builder: PromptBuilder = field(default_factory=PromptBuilder)
	video_generator: LocalVideoGenerator = field(default_factory=LocalVideoGenerator)
	frame_extractor: FrameExtractor = field(default_factory=FrameExtractor)
	stitcher: VideoStitcher = field(default_factory=VideoStitcher)

	def run(
		self,
		experiment: Experiment,
		*,
		initial_image_path: str | Path | None = None,
	) -> Path:
		"""Run the full clip loop and return the final stitched video path."""

		self._prepare_directories()

		clip_video_paths: list[Path] = []
		if self.config.continuity_mode not in {"independent", "chain"}:
			raise PipelineError("continuity_mode must be either 'independent' or 'chain'")

		try:
			if len({c.name for c in experiment.clips if c.name}) != sum(bool(c.name) for c in experiment.clips):
				raise PipelineError("Clip names must be unique")
			try:
				prepare_state_tracks(experiment)
				material_plan = compile_material_plan(experiment)
			except ValueError as exc:
				Path(self.config.experiment_output_dir, "material_plan.json").write_text(json.dumps({"status": "invalid", "error": str(exc), "scientifically_verified": False}, indent=2), encoding="utf-8")
				raise PipelineError(str(exc)) from exc
			Path(self.config.experiment_output_dir, "material_plan.json").write_text(json.dumps(material_plan, indent=2), encoding="utf-8")
			if self.config.material_plan_mode not in {"warn", "strict"}:
				raise PipelineError("material_plan_mode must be warn or strict")
			if self.config.material_review_mode not in {"observe", "strict"}:
				raise PipelineError("material_review_mode must be observe or strict")
			if material_plan["status"] == "needs_source_detail":
				issues = "; ".join(f"{row['clip']}: {issue}" for row in material_plan["clips"] for issue in row["issues"])
				if self.config.material_plan_mode == "strict":
					raise PipelineError(f"Material plan needs source detail before generation: {issues}. See material_plan.json. No GPU generation was started.")
				print(f"UNRESOLVED MATERIAL PLAN: {issues}")
			if len({c.name for c in experiment.clips}) != len(experiment.clips):
				raise PipelineError("Clip names must be unique after assigning default names")
			prompt_bundles = {}
			for index, clip in enumerate(experiment.clips):
				clip_name = clip.name or f"clip_{index + 1}"
				clip.name = clip_name
				prompt_bundles[clip_name] = self.prompt_builder.build_prompt_bundle(experiment, clip)
			# Validate references before starting the expensive GPU worker.
			seen = set()
			for clip in experiment.clips:
				if not math.isfinite(clip.motion_prompt.clip_duration_seconds) or clip.motion_prompt.clip_duration_seconds <= 0:
					raise PipelineError(f"Clip '{clip.name}' needs a finite positive duration")
				if clip.reference_mode not in {"independent", "chained"} or clip.handoff_policy not in {"none", "mid", "final", "near_end", "offset"}:
					raise PipelineError(f"Clip '{clip.name}' has an invalid reference or handoff policy")
				if clip.reference_clip and clip.reference_clip not in seen:
					raise PipelineError(f"Clip '{clip.name}' must reference an earlier clip, got '{clip.reference_clip}'")
				if clip.input_frame_path and not clip_requires_continuity(self.config.continuity_mode, clip) and not Path(clip.input_frame_path).is_file():
					raise PipelineError(f"Missing explicit reference for '{clip.name}': {clip.input_frame_path}")
				if clip.metadata.get("requires_clip_split"):
					raise PipelineError(f"Clip '{clip.name}' combines visible operations. Split reagent addition and indicator observation into separate clips before generation.")
				seen.add(clip.name)
			prompts_prepared = False
			generated_videos: dict[str, tuple[Clip, Path]] = {}
			handoff_frames: dict[tuple, Path] = {}
			for index, clip in enumerate(experiment.clips):
				clip_name = clip.name or f"clip_{index + 1}"
				clip.name = clip_name
				self._prepare_clip_directory(clip_name)
				reference_frame = None
				reference_name = ""
				mode = None
				if clip_requires_continuity(self.config.continuity_mode, clip):
					reference_name = clip.reference_clip or (experiment.clips[index - 1].name if index else "")
					if reference_name and reference_name not in generated_videos:
						raise PipelineError(
							f"Clip '{clip_name}' references unavailable clip "
							f"'{reference_name}'"
						)
					if reference_name:
						producer, source_video = generated_videos[reference_name]
						# The consumer chooses the state it needs. Legacy producer modes
						# remain the default when no consumer policy was supplied.
						mode = clip.handoff_policy if clip.handoff_policy != "none" else producer.handoff_mode
						offset = clip.handoff_offset_seconds if clip.handoff_policy != "none" else producer.handoff_offset_seconds
						key = (reference_name, mode, offset)
						if key not in handoff_frames:
							frame_path = Path(self.config.get_clip_dir(clip_name)) / f"reference_{mode}.png"
							handoff_frames[key] = self.frame_extractor.extract_handoff_frame(
								source_video, frame_path, mode=mode, offset_seconds=offset)
						reference_frame = handoff_frames[key]
						producer.extracted_frame_path = str(reference_frame)
						if not reference_frame.is_file():
							raise PipelineError(f"Missing generated-state handoff for '{clip_name}'; refusing a static-image fallback")
				if clip.reference_mode == "chained" and not reference_name:
					raise PipelineError(
						f"Clip '{clip_name}' declares chained reference_mode "
						"but has no reference_clip"
					)
				if clip.reference_mode not in {"independent", "chained"}:
					raise PipelineError(
						f"Clip '{clip_name}' has invalid reference_mode "
						f"'{clip.reference_mode}'"
					)

				prompt_bundle = prompt_bundles[clip_name]
				image_prompt_path = self._write_clip_prompts(clip_name, prompt_bundle.image_prompt, prompt_bundle.motion_prompt)

				clip_input_image_path = self._resolve_clip_input_image(
					clip=clip,
					clip_name=clip_name,
					previous_frame_path=(
						reference_frame
						if clip_requires_continuity(self.config.continuity_mode, clip)
						else None
					),
					fallback_image_path=Path(initial_image_path) if initial_image_path else None,
					image_prompt_path=image_prompt_path,
				)
				# Retain provenance even when a matching generated clip is resumed.
				provenance = {
					"reference": str(clip_input_image_path),
					"reference_sha256": hashlib.sha256(clip_input_image_path.read_bytes()).hexdigest(),
					"reference_clip": reference_name or None,
					"handoff_mode": mode,
					"state_track": clip.state_track or None,
					"starts_new_sample": clip.starts_new_sample,
					"expected_liquid_state": clip.liquid_state,
					"scientific_state_verified": False,
				}
				Path(self.config.get_clip_dir(clip_name), "reference_provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")

				output_video_path = Path(self.config.get_clip_video_file(clip_name))
				manifest_path = output_video_path.with_suffix(".generation.json")
				signature = self._generation_signature(prompt_bundle, clip_input_image_path)
				try:
					manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
				except (OSError, ValueError):
					manifest = {}
				if not isinstance(manifest, dict):
					manifest = {}
				if output_video_path.is_file() and output_video_path.stat().st_size and manifest.get("signature") == signature:
					print(f"Clip video {output_video_path} already exists. Skipping generation.")
					generated_video_path = output_video_path
				else:
					if output_video_path.exists():
						backup = output_video_path.with_name(f"{output_video_path.stem}.previous-{time.time_ns()}.mp4")
						output_video_path.rename(backup)
						if manifest_path.exists():
							manifest_path.rename(backup.with_suffix(".generation.json"))
						print(f"Changed or unverified inputs: preserving old clip as {backup}")
					if not prompts_prepared and hasattr(self.video_generator, "prepare_prompts"):
						self.video_generator.prepare_prompts([
							{"cache_key": name, "prompt": bundle.motion_prompt,
							 "negative_prompt": bundle.negative_prompt}
							for name, bundle in prompt_bundles.items()
						])
						prompts_prepared = True
					started = time.perf_counter()
					generated_video_path = self.video_generator.generate_clip_video(
						VideoGenerationRequest(
							input_image_path=clip_input_image_path,
							output_video_path=output_video_path,
							prompt_bundle=prompt_bundle,
							clip_name=clip_name,
							metadata={},
						)
					)
					manifest_path.write_text(json.dumps({
						"signature": signature, "clip": clip_name,
						"wall_seconds": time.perf_counter() - started,
						"requested_seconds": prompt_bundle.clip_duration_seconds,
						"selected_area": getattr(self.video_generator, "last_generation_area", None),
						"reference": str(clip_input_image_path),
						"reference_provenance": provenance,
					}, indent=2), encoding="utf-8")

				if "material_plan" in clip.metadata:
					status, review_path = material_review_status(generated_video_path, clip)
					print(f"Material review for {clip_name}: {status}; {review_path}")
					if status == "fail" or (self.config.material_review_mode == "strict" and status != "pass"):
						raise PipelineError(f"Material review {status} for '{clip_name}'. Inspect {review_path}; the clip is saved and will not be chained. Approved unchanged clips can be resumed without regeneration.")
				clip.output_clip_path = str(generated_video_path)
				generated_videos[clip_name] = (clip, generated_video_path)
				clip_video_paths.append(generated_video_path)

			if not clip_video_paths:
				raise PipelineError("Experiment has no clips to generate")

			final_video_path = self.stitcher.stitch_clips(clip_video_paths, self.config.stitched_video_file)
			return final_video_path
		finally:
			if hasattr(self.video_generator, "stop_daemon"):
				self.video_generator.stop_daemon()

	def _generation_signature(self, bundle, image_path: Path) -> str:
		"""Invalidate resumed clips when prompts, references, duration or settings change."""
		settings = {
			"version": 1, "prompt": asdict(bundle),
			"reference_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
			"generator": getattr(self.video_generator, "script_path", None),
			"arguments": getattr(self.video_generator, "base_arguments", []),
			"max_area": getattr(self.video_generator, "max_area", None),
			"fallback_areas": getattr(self.video_generator, "fallback_areas", ()),
		}
		return hashlib.sha256(json.dumps(settings, sort_keys=True, default=str).encode()).hexdigest()

	def _frame_is_needed(self, index: int, clip_name: str, clips: list[Clip]) -> bool:
		"""Only decode a handoff image when a later clip can consume it."""
		for later_clip in clips[index + 1 :]:
			if later_clip.reference_clip != clip_name:
				continue
			has_own_image = any(
				image_path and Path(image_path).is_file()
				for image_path in (later_clip.input_frame_path, later_clip.generated_image_path)
			)
			if not has_own_image:
				return True
		if index + 1 >= len(clips):
			return False
		next_clip = clips[index + 1]
		if not clip_requires_continuity(self.config.continuity_mode, next_clip):
			return False
		explicit_images = (next_clip.input_frame_path, next_clip.generated_image_path)
		return not any(
			image_path and Path(image_path).is_file()
			for image_path in explicit_images
		)

	def _prepare_directories(self) -> None:
		Path(self.config.experiment_output_dir).mkdir(parents=True, exist_ok=True)
		Path(self.config.motion_prompts_dir).mkdir(parents=True, exist_ok=True)
		Path(self.config.input_image_dir).mkdir(parents=True, exist_ok=True)

	def _prepare_clip_directory(self, clip_name: str) -> None:
		Path(self.config.get_clip_dir(clip_name)).mkdir(parents=True, exist_ok=True)

	def _write_clip_prompts(self, clip_name: str, image_prompt: str, motion_prompt: str) -> Path:
		image_prompt_path = Path(self.config.get_clip_dir(clip_name)) / "image_prompt.txt"
		motion_prompt_path = Path(self.config.get_clip_motion_prompt_file(clip_name))

		image_prompt_path.write_text(image_prompt.strip() + "\n", encoding="utf-8")
		motion_prompt_path.parent.mkdir(parents=True, exist_ok=True)
		motion_prompt_path.write_text(motion_prompt.strip() + "\n", encoding="utf-8")

		return image_prompt_path

	def _resolve_clip_input_image(
		self,
		*,
		clip: Clip,
		clip_name: str,
		previous_frame_path: Path | None,
		image_prompt_path: Path,
		fallback_image_path: Path | None = None,
	) -> Path:
		# A continuing sample uses its generated state, never a stale static keyframe.
		if previous_frame_path and previous_frame_path.is_file():
			return previous_frame_path
		if clip.generated_image_path:
			provided_image_path = Path(clip.generated_image_path)
			if provided_image_path.exists():
				return provided_image_path

		# Repository root (two levels up from this file)
		repo_root = Path(__file__).resolve().parent.parent
		package_root = Path(__file__).resolve().parent
		print(f"[debug] repo_root={repo_root}", file=sys.stderr)

		# Helper to check both cwd-relative and repo-root-relative locations
		def _exists_and_resolve(p: Path) -> Path | None:
			if p.exists():
				return p
			if not p.is_absolute():
				# check package-local path first (image_to_video_pipeline/<p>)
				pkg_alt = package_root / p
				if pkg_alt.exists():
					return pkg_alt
				# then check repo-root-relative path
				repo_alt = repo_root / p
				if repo_alt.exists():
					return repo_alt
			return None

		# Helper to resolve check with common image extensions (.png, .jpg, .jpeg, etc.)
		def _resolve_with_extensions(base_path_with_ext: Path) -> Path | None:
			resolved = _exists_and_resolve(base_path_with_ext)
			if resolved:
				return resolved
			stem = base_path_with_ext.stem
			parent = base_path_with_ext.parent
			for ext in ["png", "jpg", "jpeg", "PNG", "JPG", "JPEG"]:
				alt_path = parent / f"{stem}.{ext}"
				resolved = _exists_and_resolve(alt_path)
				if resolved:
					return resolved
			return None

		# Static references apply to independent scenes and intentional new samples.
		if clip.input_frame_path:
			resolved = _resolve_with_extensions(Path(clip.input_frame_path))
			if resolved:
				return resolved

		# In independent mode, reuse the verified initial reference when no
		# clip-specific keyframe is supplied.
		if fallback_image_path and fallback_image_path.exists():
			return fallback_image_path

		# 2. Candidate: clip-specific input inside the clip directory (e.g. manual image for the clip)
		expected_image_path = Path(self.config.get_clip_image_file(clip_name))
		resolved = _resolve_with_extensions(expected_image_path)
		if resolved:
			print(f"[debug] checking clip-specific image: {expected_image_path} -> resolved={resolved}", file=sys.stderr)
			return resolved

		# 3. Candidate: workspace-level input folders (e.g. "input/" or "inputs/")
		workspace_candidates = [
			Path(self.config.input_dir) / f"{self.config.experiment_name}" / f"input_image.{self.config.default_image_extension}",
			Path(self.config.input_dir) / f"input_image.{self.config.default_image_extension}",
		]
		for p in workspace_candidates:
			resolved = _resolve_with_extensions(p)
			print(f"[debug] checking workspace candidate: {p} -> resolved={resolved}", file=sys.stderr)
			if resolved:
				return resolved

		# 4. Candidate: experiment-level shared input (outputs/<experiment>/input/input_image.png)
		generic_input_image_path = Path(self.config.get_input_image_file())
		resolved = _resolve_with_extensions(generic_input_image_path)
		if not generic_input_image_path.is_absolute():
			print(f"[debug] generic alt path: {repo_root / generic_input_image_path} -> exists={(repo_root / generic_input_image_path).exists()}", file=sys.stderr)
		print(f"[debug] checking generic input image: {generic_input_image_path} -> resolved={resolved}", file=sys.stderr)
		if resolved:
			return resolved

		raise ManualImageRequiredError(
			clip_name=clip_name,
			expected_image_path=expected_image_path,
			image_prompt_path=image_prompt_path,
			message=(
				f"Manual image required for {clip_name}. "
				f"Generate an image from prompt file '{image_prompt_path}' and place it at either "
				f"'{generic_input_image_path}' or '{expected_image_path}'."
			),
		)

