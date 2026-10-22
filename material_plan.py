"""Source-defined liquid budgets and reaction extent. Expected state, never perception."""
from __future__ import annotations

import math
import re
from models import Experiment, Clip


def number(state, key, clip_name):
    if key not in state:
        return None
    value = state[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{clip_name}: {key} must be a finite nonnegative number")
    return float(value)


def compile_material_plan(experiment: Experiment) -> dict:
    """Accumulate each sample's additions/removals; retain uncertainty instead of guessing."""
    histories = {}
    records = []
    for clip in experiment.clips:
        state = clip.liquid_state
        text = f"{clip.action} {clip.state_before} {clip.state_after} {clip.motion_prompt.motion_prompt}".lower()
        liquid = bool(state) or bool(histories.get(clip.reference_clip)) or bool(re.search(r"\b(liquid|solution|water|suspension|precipitate)\b", text))
        if not liquid:
            clip.metadata.pop("material_plan", None)
            continue
        issues = []
        parent = histories.get(clip.reference_clip) if not clip.starts_new_sample else None
        before = number(state, "before_ml", clip.name)
        inherited = parent.get("expected_after_ml") if parent else None
        if before is None:
            before = inherited
        elif inherited is not None and not math.isclose(before, inherited, abs_tol=.01):
            raise ValueError(f"{clip.name}: expected starting volume resets from {inherited:g} to {before:g} mL")
        capacity = number(state, "vessel_capacity_ml", clip.name)
        working = number(state, "safe_working_volume_ml", clip.name)
        if parent:
            if capacity is not None and parent.get("vessel_capacity_ml") is not None and not math.isclose(capacity, parent["vessel_capacity_ml"], abs_tol=.01):
                raise ValueError(f"{clip.name}: capacity changed on the same vessel track")
            capacity = capacity if capacity is not None else parent.get("vessel_capacity_ml")
            working = working if working is not None else parent.get("safe_working_volume_ml")
        if capacity == 0 or working == 0 or (capacity is not None and working is not None and working > capacity):
            raise ValueError(f"{clip.name}: invalid vessel capacity/working-volume budget")
        dose = state.get("dose", {})
        if not isinstance(dose, dict):
            raise ValueError(f"{clip.name}: dose must be an object")
        kind = dose.get("kind", "unspecified")
        if kind not in {"volume", "drops", "endpoint", "none", "unspecified"}:
            raise ValueError(f"{clip.name}: invalid dose kind {kind!r}")
        addition = bool(re.search(r"\b(add|adding|pour|dispense|introduce|deliver|acidify)\b", clip.action.lower() or clip.motion_prompt.motion_prompt.lower()))
        # Source facts also identify additions phrased without an 'add' verb.
        # An unknown dose must propagate uncertainty, never become zero mL.
        addition = addition or any(f.get("role") in {"addition", "transfer"}
                                   for f in clip.metadata.get("quantities", []))
        addition = addition or kind in {"volume", "drops", "endpoint"} or "added_ml" in state
        added = number(state, "added_ml", clip.name)
        bound = None
        if kind == "volume":
            if dose.get("unit", "mL") != "mL":
                raise ValueError(f"{clip.name}: convert sourced dose volumes to mL explicitly")
            value = number(dose, "value", clip.name)
            if value is None:
                raise ValueError(f"{clip.name}: volume dose requires value")
            if added is not None and not math.isclose(added, value, abs_tol=.01):
                raise ValueError(f"{clip.name}: dose disagrees with added_ml")
            added = bound = value
        elif kind == "drops":
            count = dose.get("count")
            if isinstance(count, bool) or not isinstance(count, int) or count < 1:
                raise ValueError(f"{clip.name}: drops dose needs a positive integer count")
            drop_ml = number(dose, "drop_volume_ml", clip.name)
            if drop_ml is not None:
                if drop_ml <= 0:
                    raise ValueError(f"{clip.name}: calibrated drop volume must be positive")
                calculated = count * drop_ml
                if added is not None and not math.isclose(added, calculated, abs_tol=.01):
                    raise ValueError(f"{clip.name}: drop calibration disagrees with added_ml")
                added = bound = calculated
            else:
                bound = number(dose, "max_added_ml", clip.name)
        elif kind == "endpoint":
            if not str(dose.get("endpoint", "")).strip():
                raise ValueError(f"{clip.name}: endpoint dose requires the source-defined endpoint")
            bound = number(dose, "max_added_ml", clip.name)
            if bound is None:
                issues.append("Endpoint addition has no sourced upper volume bound; cumulative capacity cannot be established")
        elif kind == "none":
            if added not in (None, 0) or addition:
                raise ValueError(f"{clip.name}: dose 'none' conflicts with an addition")
            added = bound = 0.0
        if added is not None:
            if bound is not None and added > bound + .01:
                raise ValueError(f"{clip.name}: addition exceeds its documented dose bound")
            bound = added
        if addition and kind == "unspecified" and added is None:
            issues.append("Addition has no documented volume, drop count or bounded endpoint; equal doses must not be assumed")
        # A removal requires its own explicit amount. Absence of a removal action
        # means zero removed, rather than silently draining the vessel at a boundary.
        removal = bool(re.search(r"\b(withdraw|remove|decant|evaporate|pipette out)\b", clip.action.lower()))
        removed = number(state, "removed_ml", clip.name)
        if removed is None and not removal:
            removed = 0.0
        if removed is None:
            issues.append("Removal/evaporation amount is unknown")
        if not addition and added is None:
            added = bound = 0.0
        after = number(state, "after_ml", clip.name)
        volume_model = state.get("volume_model", "additive_budget")
        if volume_model not in {"additive_budget", "source_defined"}:
            raise ValueError(f"{clip.name}: invalid volume_model")
        if volume_model == "source_defined" and after is None:
            issues.append("Source-defined volume model needs a documented final volume")
        # Added liquid volumes form a nominal planning budget. Real mixing,
        # dissolution and reaction volumes need source-defined endpoints when
        # contraction, expansion or displacement matters.
        calculated = before + added - removed if volume_model == "additive_budget" and None not in (before, added, removed) else None
        if calculated is not None:
            if calculated < -.01:
                raise ValueError(f"{clip.name}: removed volume exceeds available contents")
            if after is not None and not math.isclose(after, calculated, abs_tol=.01):
                raise ValueError(f"{clip.name}: declared final volume violates cumulative volume balance")
            after = max(0.0, calculated)
        upper_before = before if before is not None else parent.get("upper_after_ml") if parent else None
        upper_after = upper_before + bound - removed if None not in (upper_before, bound, removed) else None
        if after is not None:
            upper_after = max(after, upper_after) if upper_after is not None else after
        if before is None and (addition or removal):
            issues.append("Starting liquid volume is unknown; inspect/calibrate the reference rather than inventing mL")
        if addition and upper_after is None:
            issues.append("Cumulative final volume cannot be bounded")
        limit = working if working is not None else capacity
        for label, value in (("starting", before), ("final", upper_after)):
            if limit is not None and value is not None and value > limit + .01:
                raise ValueError(f"{clip.name}: {label} volume {value:g} mL exceeds sourced vessel budget {limit:g} mL; revise the source plan, use a larger vessel, or explicitly create an allowed aliquot")
        if addition and limit is None:
            issues.append("Vessel capacity/working-volume limit is unspecified")
        if kind in {"volume", "drops", "endpoint"} and not str(dose.get("source", "")).strip():
            issues.append("Dose has no source citation")
        scope = state.get("reaction_scope", "source_defined")
        if scope not in {"whole_contents", "localized", "interface", "stratified", "source_defined"}:
            raise ValueError(f"{clip.name}: invalid reaction_scope")
        clarity = state.get("after_clarity", "source_defined")
        if clarity not in {"clear", "cloudy", "opaque", "source_defined"}:
            raise ValueError(f"{clip.name}: invalid after_clarity")
        if state.get("after_solid", "source_defined") not in {"none", "present", "source_defined"}:
            raise ValueError(f"{clip.name}: invalid after_solid")
        minimum = number(state, "minimum_visible_seconds", clip.name)
        if minimum is not None and minimum > clip.motion_prompt.clip_duration_seconds:
            raise ValueError(f"{clip.name}: the documented operation needs at least {minimum:g}s; the requested clip is too short. Plan a separate observation or a validated longer profile")
        record = {"clip": clip.name, "reference_clip": clip.reference_clip or None,
                  "state_track": clip.state_track or None, "expected_before_ml": before,
                  "expected_added_ml": added, "expected_removed_ml": removed,
                  "expected_after_ml": after, "upper_after_ml": upper_after,
                  "vessel_capacity_ml": capacity, "safe_working_volume_ml": working,
                  "dose": dose, "reaction_scope": scope, "after_clarity": clarity,
                  "issues": issues, "observed_or_verified": False}
        clip.metadata["material_plan"] = record
        histories[clip.name] = record
        records.append(record)
    return {"schema_version": 1, "status": "needs_source_detail" if any(r["issues"] for r in records) else "numerically_consistent",
            "scientifically_verified": False, "clips": records}


def material_constraints(clip: Clip) -> list[str]:
    state = clip.liquid_state
    plan = clip.metadata.get("material_plan", {})
    dose = state.get("dose", {})
    parts = []
    if dose.get("kind") == "drops":
        parts.append(f"Deliver exactly {dose['count']} separate drops, then stop dispensing; do not pour or empty the dropper")
    elif dose.get("kind") == "volume":
        parts.append(f"Deliver only the documented {dose['value']:g} mL, then stop; clip duration does not determine dose")
    elif dose.get("kind") == "endpoint":
        parts.append(f"Add incrementally only until {dose['endpoint']}, then stop; do not continue filling while waiting for the reaction")
    if plan.get("expected_before_ml") is not None:
        parts.append(f"Retain the existing {plan['expected_before_ml']:g} mL; no emptying or reset")
    if plan.get("expected_after_ml") is not None:
        parts.append(f"End at {plan['expected_after_ml']:g} mL total, not the dose volume")
    if state.get("after_clarity") == "clear":
        parts.append("Final liquid is optically transparent throughout: the background is visible through it")
    if state.get("after_colour"):
        parts.append(f"Final liquid colour: {state['after_colour']}")
    if state.get("reaction_scope") == "whole_contents":
        parts.append("The documented transformation reaches all original contents from base to meniscus; added liquid does not lift an unchanged plug or cap above a transformed lower half")
        if state.get("after_solid") == "none":
            parts.append("At the endpoint the original solid has dissolved throughout; the liquid forms one continuous transparent region")
    elif state.get("reaction_scope") in {"localized", "interface", "stratified"}:
        parts.append(f"Preserve the source-defined {state['reaction_scope']} reaction; do not force every experiment to become homogeneous")
    if state.get("mixing_instruction"):
        parts.append(f"Source-defined mixing: {state['mixing_instruction']}")
    return parts
