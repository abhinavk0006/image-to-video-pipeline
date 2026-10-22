"""Build text prompts from the shared experiment models.

This module is responsible for turning an `Experiment` and one of its `Clip`
definitions into human-readable prompt text that downstream image and video
generation steps can consume.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re

from models import Clip, Experiment, PromptBundle, Scene
from material_state import liquid_constraints, tool_constraint
from material_plan import material_constraints


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
			"Start from the supplied image. Preserve vessel identity, count, shape and position; contents change only through the stated operation",
			f"Target object: {clip.target_object}" if clip.target_object else "",
			f"Keep unchanged: {_join_phrases(clip.preserve_objects)}" if clip.preserve_objects else "",
			f"Quantity limit: {clip.quantity_constraint or 'use documented amounts only; keep the liquid below the vessel rim'}",
			f"Observable result: {clip.observable_target}" if clip.observable_target and clip.observable_target != clip.state_after else "",
			f"Pacing: {clip.action_pacing or 'one slow, deliberate action, then hold the result still; no time lapse'}",
			"Perform the operation once on its named target; stop delivery before holding the result",
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
		# Put the action, expected appearance and contact rule ahead of general
		# styling. CFG=1 Lightning ignores the negative prompt during denoising.
		critical = [
			f"Action: {clip.action}" if clip.action else action_text,
			*material_constraints(clip),
			f"Required result: {clip.state_after}" if clip.state_after else "",
			*liquid_constraints(clip),
			tool_constraint(clip),
		]
		return _join_sentences([*critical, *grounding,
			f"Starting state: {clip.state_before}" if clip.state_before else "",
			motion.moving_object, motion.motion,
			f"Direction: {motion.direction}" if motion.direction else "",
			motion.constraints,
			motion.stop_condition if motion.stop_condition != clip.state_after else ""])

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
				"state_track": clip.state_track,
				"starts_new_sample": clip.starts_new_sample,
				"liquid_state": clip.liquid_state,
				"tool_contact_policy": clip.tool_contact_policy,
			},
		)


def build_prompt_bundle(experiment: Experiment, clip: Clip) -> PromptBundle:
	"""Convenience wrapper for the default prompt builder."""

	return PromptBuilder().build_prompt_bundle(experiment, clip)
