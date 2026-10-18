"""Source-declared sample continuity and liquid/tool constraints, not visual verification."""
from __future__ import annotations

import math
import re
from models import Experiment, Clip


def prepare_state_tracks(experiment: Experiment) -> None:
    """Chain the latest generated state of each declared sample, even across interleaving."""
    latest: dict[str, Clip] = {}
    for index, clip in enumerate(experiment.clips):
        clip.name = clip.name or f"clip_{index + 1}"
        liquid_constraints(clip)  # Reject impossible numeric contracts before GPU startup.
        tool_constraint(clip)
        if clip.starts_new_sample:
            if clip.reference_clip or clip.reference_mode == "chained" or clip.continuity_required:
                raise ValueError(f"{clip.name}: a new sample cannot also request a generated-state reference")
            if not (clip.input_frame_path or clip.generated_image_path):
                raise ValueError(f"{clip.name}: a new sample needs its own reference image")
            clip.continuity_required = False
            clip.reference_policy = "state_keyframe"
        if not clip.state_track:
            continue
        parent = latest.get(clip.state_track)
        if parent and not clip.starts_new_sample:
            if clip.reference_clip and clip.reference_clip != parent.name:
                raise ValueError(f"{clip.name}: track {clip.state_track!r} must reference its latest state {parent.name!r}; declare a new sample for an intentional reset")
            clip.reference_clip = parent.name
            clip.reference_mode = "chained"
            clip.continuity_required = True
            clip.needs_reference_image = False
            if clip.handoff_policy == "none":
                clip.handoff_policy = "final"
            if "after_ml" in parent.liquid_state and "before_ml" in clip.liquid_state:
                if not math.isclose(parent.liquid_state["after_ml"], clip.liquid_state["before_ml"], abs_tol=0.01):
                    raise ValueError(f"{clip.name}: starting volume disagrees with the preceding state on this sample track")
        elif not clip.reference_clip and not clip.starts_new_sample:
            if clip.reference_mode == "chained":
                raise ValueError(f"{clip.name}: the first clip on a new track needs an explicit reference, not an implicit preceding sample")
            if index and not (clip.input_frame_path or clip.generated_image_path):
                raise ValueError(f"{clip.name}: a new state track needs its own starting image")
            clip.continuity_required = False
            clip.reference_mode = "independent"
            clip.reference_policy = "state_keyframe"
        latest[clip.state_track] = clip


def liquid_constraints(clip: Clip) -> list[str]:
    """Only constrain colours/volumes supplied by the source; never infer a reaction."""
    state = clip.liquid_state
    parts = []
    for key in ("before_ml", "added_ml", "removed_ml", "after_ml", "max_fill_fraction"):
        value = state.get(key)
        if key in state and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0):
            raise ValueError(f"{clip.name}: liquid_state.{key} must be a finite nonnegative number")
    cap = state.get("max_fill_fraction")
    if cap is not None and not 0 < cap < 1:
        raise ValueError(f"{clip.name}: max_fill_fraction must be between zero and one")
    if all(key in state for key in ("before_ml", "added_ml", "removed_ml", "after_ml")):
        expected = state["before_ml"] + state["added_ml"] - state["removed_ml"]
        if not math.isclose(expected, state["after_ml"], abs_tol=0.01):
            raise ValueError(f"{clip.name}: liquid volume balance is inconsistent")
    for key, label in (("before_appearance", "Initial liquid appearance"), ("after_appearance", "Required final liquid appearance"), ("solid_change", "Solid behaviour")):
        if state.get(key):
            parts.append(f"{label}: {state[key]}")
    for key, label in (("before_ml", "Starting volume"), ("added_ml", "Added volume"), ("removed_ml", "Removed volume"), ("after_ml", "Final volume")):
        if key in state:
            parts.append(f"{label}: {state[key]:g} mL")
    if cap is not None:
        parts.append(f"Keep the meniscus below {cap * 100:g}% of the vessel's internal height")
    if clip.state_track or clip.continuity_required or clip.reference_clip:
        parts.append("Begin at the exact meniscus and material state visible in the reference; retain that liquid rather than resetting the tube")
    if state.get("level_change"):
        allowed = {"increase", "decrease", "unchanged", "source_defined"}
        if state["level_change"] not in allowed:
            raise ValueError(f"{clip.name}: unknown level_change")
        parts.append(f"Liquid level change during this operation: {state['level_change']}; continuous gradual movement, no sudden refill, emptying or jump")
    return parts


def tool_constraint(clip: Clip) -> str:
    policy = clip.tool_contact_policy
    text = f"{clip.action} {clip.motion_prompt.motion_prompt} {clip.motion_prompt.motion} {clip.motion_prompt.moving_object}".lower()
    delivery = re.search(r"\b(add|adding|dispense|dispensing|deliver|delivering)\b", text)
    dropper = re.search(r"\b(dropper|dropwise|droppers)\b", text)
    if policy not in {"auto", "above_rim", "above_liquid", "source_defined", "immersion_required"}:
        raise ValueError(f"{clip.name}: invalid tool_contact_policy {policy!r}")
    if policy == "auto":
        immersion = re.search(r"\b(aspirate|aspirating|withdraw|sampling|submerged|immerse|immersion|subsurface)\b", text)
        policy = "above_rim" if delivery and dropper and not immersion else "source_defined"
    if policy == "above_rim":
        rule = "Keep the dispensing tip outside and above the vessel opening, with a visible air gap; delivered liquid crosses the gap into the existing vessel. The tip never enters the tube or touches the liquid"
        if delivery and dropper:
            rule += ". Release individual drops one at a time; the level rises only by the delivered amount, never an unexplained large refill"
        return rule
    if policy == "above_liquid":
        return "Keep the dispensing tip above the highest liquid surface throughout delivery, with a visible air gap; do not submerge it"
    if policy == "immersion_required":
        return "Use only the tool immersion explicitly required by the source procedure"
    return ""
