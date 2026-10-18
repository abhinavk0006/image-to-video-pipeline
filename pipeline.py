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
	if clip.continuity_required is not None:
		return clip.continuity_required
	if clip.reference_clip or clip.reference_mode == "chained":
		return True
	return config_mode == "chain" and clip.reference_policy not in {"independent", "state_keyframe"}


@dataclass(slots=True)
class ExperimentPipeline:
	"""Coordinates clip-by-clip generation from structured experiment data."""

	config: PipelineConfig = field(default_factory=PipelineConfig)
	prompt_builder: PromptBuilder = field(default_factory=PromptBuilder)
	video_generator: LocalVideoGenerator = field(default_factory=LocalVideoGenerator)
	frame_extractor: FrameExtractor = field(default_factory=FrameExtractor)
	stitcher: VideoStitcher = field(default_factory=VideoStitcher)
	_repairs_used: int = field(default=0, init=False)

	def run(
		self,
		experiment: Experiment,
		*,
		initial_image_path: str | Path | None = None,
	) -> Path:
		"""Run the full clip loop and return the final stitched video path."""

		self._prepare_directories()
		self._repairs_used = 0
		if self.config.science_mode not in {'off', 'observe', 'strict'} or self.config.science_repair_budget < 0:
			raise PipelineError('Invalid science mode or repair budget')

		clip_video_paths: list[Path] = []
		previous_frame_path = Path(initial_image_path) if initial_image_path else None
		if self.config.continuity_mode not in {"independent", "chain"}:
			raise PipelineError("continuity_mode must be either 'independent' or 'chain'")

		try:
			if len({c.name for c in experiment.clips if c.name}) != sum(bool(c.name) for c in experiment.clips):
				raise PipelineError("Clip names must be unique")
			prompt_bundles = {}
			for index, clip in enumerate(experiment.clips):
				clip_name = clip.name or f"clip_{index + 1}"
				clip.name = clip_name
				prompt_bundles[clip_name] = self.prompt_builder.build_prompt_bundle(experiment, clip)
			# Validate references before starting the expensive GPU worker.
			seen = set()
			for clip in experiment.clips:
				if self.config.science_mode != 'off':
					from science_evidence import contract_for
					contract_for(clip)
				if not math.isfinite(clip.motion_prompt.clip_duration_seconds) or clip.motion_prompt.clip_duration_seconds <= 0:
					raise PipelineError(f"Clip '{clip.name}' needs a finite positive duration")
				if clip.reference_mode not in {"independent", "chained"} or clip.handoff_policy not in {"none", "mid", "final", "near_end", "offset", "reviewed"}:
					raise PipelineError(f"Clip '{clip.name}' has an invalid reference or handoff policy")
				if clip.reference_clip and clip.reference_clip not in seen:
					raise PipelineError(f"Clip '{clip.name}' must reference an earlier clip, got '{clip.reference_clip}'")
				if clip.input_frame_path and not Path(clip.input_frame_path).is_file():
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
				if clip.metadata.get('render_mode') == 'circuit_diagram':
					from circuit_renderer import render_reference
					clip.input_frame_path = str(render_reference(clip.metadata['circuit'],
						Path(self.config.get_clip_dir(clip_name)) / 'circuit_reference'))
				reference_frame = previous_frame_path
				explicit_input_available = any(
					image_path and Path(image_path).is_file()
					for image_path in (clip.input_frame_path, clip.generated_image_path)
				)
				if clip_requires_continuity(self.config.continuity_mode, clip) and not explicit_input_available:
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
						key = (reference_name, mode, offset, tuple(clip.metadata.get('handoff_requirements', [])))
						if key not in handoff_frames:
							frame_path = Path(self.config.get_clip_dir(clip_name)) / f"reference_{mode}.png"
							if mode == 'reviewed':
								from science_evidence import select_reviewed_handoff
								try:
									handoff_frames[key] = select_reviewed_handoff(source_video,
										Path(self.config.get_clip_dir(reference_name)) / 'science/review.json',
										clip.metadata.get('handoff_requirements', []), frame_path)
								except (OSError, ValueError) as error:
									raise PipelineError(f'Reviewed handoff unavailable for {clip_name}: {error}') from error
							else:
								handoff_frames[key] = self.frame_extractor.extract_handoff_frame(
									source_video, frame_path, mode=mode, offset_seconds=offset)
						reference_frame = handoff_frames[key]
						producer.extracted_frame_path = str(reference_frame)
				if clip.reference_mode == "chained" and not clip.reference_clip and not explicit_input_available:
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

				output_video_path = Path(self.config.get_clip_video_file(clip_name))
				if self.config.science_mode != 'off':
					from science_evidence import visibility_preflight, write_json
					from PIL import Image
					with Image.open(clip_input_image_path) as input_image:
						width, height = input_image.size
					area = getattr(self.video_generator, 'max_area', None)
					projected = None
					if area and clip.metadata.get('render_mode') != 'circuit_diagram':
						aspect = height / width
						projected = (max(16, int(math.sqrt(area/aspect)//16)*16), max(16, int(math.sqrt(area*aspect)//16)*16))
					visibility = visibility_preflight(clip_input_image_path, clip.metadata.get('visibility_targets', []), output_size=projected)
					write_json(Path(self.config.get_clip_dir(clip_name)) / 'science/visibility.json', visibility)
					if self.config.science_mode == 'strict' and visibility['status'] != 'pass':
						raise PipelineError(f'Visibility preflight {visibility["status"]} for {clip_name}; annotate target regions and inspect science/visibility.json')
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
					if clip.metadata.get('render_mode') != 'circuit_diagram' and not prompts_prepared and hasattr(self.video_generator, "prepare_prompts"):
						self.video_generator.prepare_prompts([
							{"cache_key": name, "prompt": bundle.motion_prompt,
							 "negative_prompt": bundle.negative_prompt}
							for name, bundle in prompt_bundles.items() if next(c for c in experiment.clips if c.name == name).metadata.get('render_mode') != 'circuit_diagram'
						])
						prompts_prepared = True
					started = time.perf_counter()
					if clip.metadata.get('render_mode') == 'circuit_diagram':
						from circuit_renderer import render_animation
						generated_video_path = render_animation(clip.metadata['circuit'],
							clip.metadata.get('diagram_timeline', []), output_video_path,
							seconds=prompt_bundle.clip_duration_seconds)
					else:
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
					}, indent=2), encoding="utf-8")

				clip.output_clip_path = str(generated_video_path)
				if self.config.science_mode != 'off':
					generated_video_path = self._review_and_repair(clip, generated_video_path, prompt_bundle, clip_input_image_path)
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
			"version": 2, "prompt": {key: value for key, value in asdict(bundle).items() if key != 'metadata'},
			"reference_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
			"generator": getattr(self.video_generator, "script_path", None),
			"arguments": getattr(self.video_generator, "base_arguments", []),
			"max_area": getattr(self.video_generator, "max_area", None),
			"fallback_areas": getattr(self.video_generator, "fallback_areas", ()),
			"rendering": bundle.metadata.get('render_mode', 'wan'),
			"circuit": bundle.metadata.get('circuit'),
			"diagram_timeline": bundle.metadata.get('diagram_timeline'),
		}
		return hashlib.sha256(json.dumps(settings, sort_keys=True, default=str).encode()).hexdigest()

	def _review_and_repair(self, clip, video, bundle, image):
		from science_evidence import review_clip, write_json, sha256, contract_for
		directory = Path(self.config.get_clip_dir(clip.name)) / 'science'
		while True:
			result, review = review_clip(clip, video, directory, self.config.science_verifier_script)
			print(f'Scientific review {clip.name}: {result}; evidence: {directory}')
			if result != 'rejected' or not self.config.science_verifier_script or self._repairs_used >= self.config.science_repair_budget or clip.metadata.get('render_mode') == 'circuit_diagram':
				break
			self._repairs_used += 1
			seed = int.from_bytes(hashlib.sha256(f'{clip.name}:repair:{sha256(video)}:{self._repairs_used}'.encode()).digest()[:4], 'big')
			backup = video.with_name(f'{video.stem}.rejected-{time.time_ns()}.mp4')
			video.rename(backup)
			write_json(backup.with_suffix('.science-evidence.json'), {'contract': contract_for(clip),
				'review': {**review, 'video': str(backup.resolve())}})
			manifest = video.with_suffix('.generation.json')
			if manifest.exists():
				manifest.rename(backup.with_suffix('.generation.json'))
			self.video_generator.prepare_prompts([{'cache_key': clip.name,
				'prompt': bundle.motion_prompt, 'negative_prompt': bundle.negative_prompt}])
			started = time.perf_counter()
			video = self.video_generator.generate_clip_video(VideoGenerationRequest(
				input_image_path=image, output_video_path=video, prompt_bundle=bundle,
				clip_name=clip.name, metadata={'seed': str(seed)}))
			write_json(manifest, {'signature': self._generation_signature(bundle, image),
				'clip': clip.name, 'repair_seed': seed, 'wall_seconds': time.perf_counter()-started,
				'requested_seconds': bundle.clip_duration_seconds, 'reference': str(image),
				'selected_area': getattr(self.video_generator, 'last_generation_area', None)})
		if self.config.science_mode == 'strict' and result != 'accepted':
			raise PipelineError(f'Scientific review {result} for {clip.name}; complete {directory / "review.json"} and rerun')
		return video

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

		# Explicit per-clip references always win, regardless of continuity mode.
		if clip.input_frame_path:
			resolved = _resolve_with_extensions(Path(clip.input_frame_path))
			if resolved:
				return resolved

		# The caller only supplies a previous frame for explicitly continuous clips.
		if previous_frame_path and previous_frame_path.exists():
			return previous_frame_path

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

