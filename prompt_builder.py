"""Build text prompts from the shared experiment models.

This module is responsible for turning an `Experiment` and one of its `Clip`
definitions into human-readable prompt text that downstream image and video
generation steps can consume.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re

from models import Clip, Experiment, PromptBundle, Scene


def _clean_parts(parts: list[str]) -> list[str]:
	return [part.strip() for part in parts if part and part.strip()]


def _join_sentences(parts: list[str]) -> str:
	return ". ".join(_clean_parts(parts))


def _join_phrases(parts: list[str]) -> str:
	return ", ".join(_clean_parts(parts))


@dataclass(slots=True)
class PromptBuilder:
	"""Constructs image and motion prompts for a single clip."""

	image_suffix: str = "high quality educational science illustration, realistic composition"
	motion_suffix: str = "keep motion physically plausible and visually clear"
	metadata: dict[str, str] = field(default_factory=dict)

	def build_image_prompt(self, experiment: Experiment, scene: Scene, clip: Clip) -> str:
		parts = [
			f"Create a {experiment.name} scene",
			experiment.description,
			scene.scene,
			f"Camera: {scene.camera}" if scene.camera else "",
			f"Lighting: {scene.lighting}" if scene.lighting else "",
			f"Objects: {_join_phrases(scene.objects)}" if scene.objects else "",
			f"Background: {scene.background}" if scene.background else "",
			self.metadata.get("image_style", ""),
			self.image_suffix,
		]
		prompt = _join_sentences(parts)

		if clip.image_prompt or clip.motion_prompt.image_prompt:
			prompt = _join_sentences([
				prompt,
				clip.image_prompt,
				clip.motion_prompt.image_prompt,
			])

		return prompt

	def build_motion_prompt(self, experiment: Experiment, clip: Clip) -> str:
		motion = clip.motion_prompt
		# A complete clip prompt is the action source of truth. Appending a second
		# paraphrase of its state/action often made Wan repeat or contradict itself.
		if motion.motion_prompt.strip() and not clip.action.strip():
			action_text = motion.motion_prompt.strip()
		else:
			action_text = _join_sentences([
				f"Animate one step of {experiment.name}",
				f"Starting state: {clip.state_before}" if clip.state_before else "",
				f"Action: {clip.action}" if clip.action else "",
				motion.moving_object,
				motion.motion,
				f"Direction: {motion.direction}" if motion.direction else "",
				f"Stop when: {motion.stop_condition}" if motion.stop_condition else "",
				f"Required ending state: {clip.state_after}" if clip.state_after else "",
			])

		# These fields encode visual grounding that a prose prompt can otherwise
		# leave ambiguous: which existing object moves, what stays put, and where
		# the observable change must appear.
		grounding = [
			"Start from the supplied image. Keep the existing vessels in the same positions, with the same count, shape and contents",
			f"Target object: {clip.target_object}" if clip.target_object else "",
			f"Keep unchanged: {_join_phrases(clip.preserve_objects)}" if clip.preserve_objects else "",
			f"Quantity limit: {clip.quantity_constraint or 'use documented amounts only; keep the liquid below the vessel rim'}",
			f"Observable result: {clip.observable_target}" if clip.observable_target else "",
			f"Pacing: {clip.action_pacing or 'one slow, deliberate action, then hold the result still; no time lapse'}",
			"Only the tool named in this step may enter the frame; use the existing vessel throughout",
			"Perform this action once. Show the documented result only on its named physical target",
			f"Stop when: {motion.stop_condition}" if motion.stop_condition else "",
			f"Additional constraints: {motion.constraints}" if motion.constraints else "",
			f"Camera: {clip.image_state.camera}" if clip.image_state.camera else "",
			self.metadata.get("motion_style", ""),
			self.motion_suffix,
		]
		text = " ".join([clip.target_object, clip.action, clip.state_after, action_text]).lower()
		if "litmus" in text:
			# This specifies where an explicitly requested observation belongs;
			# it never infers a reagent, reaction, or expected colour.
			grounding.insert(1, "Keep the litmus paper clearly visible. Any documented litmus colour change occurs on the paper only; the liquid retains its documented colour")
			if re.search(r"\b(mouth|opening|gas|above)\b", text):
				grounding.insert(2, "Hold the paper outside the liquid at the existing test tube opening. This shot shows the paper observation only; reagent addition belongs in a separate shot")
		# Put spatial constraints before the action so long source text cannot
		# push them out of the text encoder's token window. Negative prompts
		# have no denoising effect in the CFG=1 Lightning path.
		return _join_sentences([*grounding, action_text,
			f"End state: {clip.state_after}" if clip.state_after else ""])

	def build_prompt_bundle(self, experiment: Experiment, clip: Clip) -> PromptBundle:
		"""Create a normalized prompt bundle for downstream modules."""

		image_prompt = self.build_image_prompt(experiment, clip.image_state, clip)
		motion_prompt = self.build_motion_prompt(experiment, clip)

		return PromptBundle(
			image_prompt=image_prompt,
			motion_prompt=motion_prompt,
			moving_object=clip.motion_prompt.moving_object,
			direction=clip.motion_prompt.direction,
			motion=clip.motion_prompt.motion,
			constraints=clip.motion_prompt.constraints,
			stop_condition=clip.motion_prompt.stop_condition,
			clip_duration_seconds=clip.motion_prompt.clip_duration_seconds,
			negative_prompt=clip.motion_prompt.negative_prompt,
			metadata={
				**clip.motion_prompt.metadata,
				**clip.metadata,
				**self.metadata,
			},
		)


def build_prompt_bundle(experiment: Experiment, clip: Clip) -> PromptBundle:
	"""Convenience wrapper for the default prompt builder."""

	return PromptBuilder().build_prompt_bundle(experiment, clip)
