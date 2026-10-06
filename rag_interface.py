"""RAG lookup and Qwen fallback for structured experiment video plans."""

from __future__ import annotations

import json
import re
from difflib import get_close_matches
from pathlib import Path
from typing import Any, Mapping

KNOWLEDGE_DIR = Path("knowledge")
_DURATION_KEYS = ("duration_seconds", "estimated_duration_seconds", "clip_duration_seconds")


def _load_index() -> dict[str, Path]:
    index: dict[str, Path] = {}
    for file_path in KNOWLEDGE_DIR.glob("*.json"):
        try:
            data = json.loads(file_path.read_text(encoding="utf-8"))
            key = str(data.get("experiment_name", data.get("name", file_path.stem))).lower()
            index[key] = file_path
            for clip in data.get("clips", []):
                step = str(clip.get("step", clip.get("name", ""))).lower()
                if step:
                    index[step] = file_path
        except (OSError, json.JSONDecodeError, AttributeError) as error:
            print(f"[rag] unable to index {file_path}: {error}")
    return index


def lookup(query: str) -> dict[str, Any] | None:
    """Return the closest local knowledge record, or ``None`` on a miss."""
    index = _load_index()
    normalized_query = query.lower().strip()
    if normalized_query in index:
        return json.loads(index[normalized_query].read_text(encoding="utf-8"))
    close = get_close_matches(normalized_query, index.keys(), n=1, cutoff=0.5)
    if close:
        print(f"[rag] fuzzy match: '{normalized_query}' -> '{close[0]}'")
        return json.loads(index[close[0]].read_text(encoding="utf-8"))
    return None


def _text(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _first_text(data: Mapping[str, Any], *keys: str) -> str:
    for key in keys:
        value = _text(data.get(key))
        if value:
            return value
    return ""


def _duration(data: Mapping[str, Any]) -> float:
    for key in _DURATION_KEYS:
        value = data.get(key)
        if value is not None and value != "":
            try:
                duration = float(value)
            except (TypeError, ValueError) as error:
                raise ValueError(f"clip duration must be numeric, got {value!r}") from error
            if duration <= 0:
                raise ValueError("clip duration must be greater than zero")
            return duration
    return 1.0


def _normalize_clip(raw_clip: Mapping[str, Any], index: int) -> dict[str, Any]:
    """Adapt Qwen, dataset, and legacy clip shapes without dropping metadata."""
    clip = dict(raw_clip)
    prompt = dict(clip.get("prompt_bundle") or clip.get("motion_prompt") or {})
    scene = dict(clip.get("scene") or clip.get("image_state") or {})

    instruction = _first_text(clip, "procedure_text", "instruction", "step", "action")
    observation = _first_text(
        clip, "observation_text", "observation", "state_after", "stop_condition"
    )
    duration = _duration({**prompt, **clip})
    clip["name"] = _first_text(clip, "name", "clip_id") or f"clip_{index}"
    clip["step"] = _first_text(clip, "step", "name") or f"Step {index}"
    clip["procedure_text"] = instruction
    clip["observation_text"] = observation
    clip["instruction"] = instruction
    clip["observation"] = observation
    clip["state_before"] = _first_text(clip, "state_before")
    clip["action"] = _first_text(clip, "action", "instruction")
    clip["state_after"] = _first_text(clip, "state_after", "observation")
    clip["duration_seconds"] = duration
    clip["estimated_duration_seconds"] = clip.get("estimated_duration_seconds", duration)
    clip["reference_policy"] = _first_text(
        clip, "reference_policy", "reference_mode"
    ) or ("previous_state_keyframe" if clip.get("reference_clip") else "state_keyframe")
    clip["reference_clip"] = _first_text(clip, "reference_clip")
    clip["reference_mode"] = _first_text(clip, "reference_mode") or (
        "chained" if clip["reference_clip"] else "independent"
    )
    clip["continuity_required"] = bool(
        clip.get("continuity_required", bool(clip["reference_clip"]))
    )
    clip["handoff_policy"] = _first_text(clip, "handoff_policy", "handoff_mode")
    if not clip["handoff_policy"]:
        clip["handoff_policy"] = "near_end" if clip["reference_clip"] else "none"
    clip["handoff_mode"] = _first_text(clip, "handoff_mode") or "near_end"
    clip["handoff_entity"] = _first_text(clip, "handoff_entity")
    clip["visual_priority"] = _first_text(clip, "visual_priority") or "normal"

    prompt["clip_duration_seconds"] = duration
    prompt.setdefault("motion_prompt", _first_text(clip, "motion_prompt", "action", "instruction"))
    prompt.setdefault("negative_prompt", _first_text(clip, "negative_prompt"))
    clip["prompt_bundle"] = prompt
    clip["scene"] = scene
    return clip


def normalize_experiment_plan(data: Mapping[str, Any], query: str = "") -> dict[str, Any]:
    """Validate and normalize a plan into the pipeline's experiment/video contract."""
    if not isinstance(data, Mapping):
        raise ValueError("RAG/Qwen response must be a JSON object")

    plan = dict(data)
    name = _first_text(plan, "name", "experiment_name", "title") or query.strip()
    if not name:
        raise ValueError("experiment plan is missing an experiment name")

    raw_clips = plan.get("clips")
    if not isinstance(raw_clips, list) or not raw_clips:
        raw_clips = plan.get("procedure_steps")
    if not isinstance(raw_clips, list) or not raw_clips:
        raise ValueError("experiment plan must contain a non-empty clips or procedure_steps list")

    clips = []
    procedure_steps = []
    for index, raw_clip in enumerate(raw_clips, start=1):
        if not isinstance(raw_clip, Mapping):
            raise ValueError(f"experiment step {index} must be a JSON object")
        clip = _normalize_clip(raw_clip, index)
        clips.append(clip)
        procedure_steps.append(
            {
                "step_id": index,
                "instruction": clip["procedure_text"],
                "observation": clip["observation_text"],
                "state_before": clip["state_before"],
                "action": clip["action"],
                "state_after": clip["state_after"],
            }
        )

    plan["name"] = name
    plan["experiment_name"] = _first_text(plan, "experiment_name") or name
    plan["clips"] = clips
    plan["procedure_steps"] = plan.get("procedure_steps") if isinstance(plan.get("procedure_steps"), list) else procedure_steps
    plan["procedure_text"] = _first_text(plan, "procedure_text") or "\n".join(
        f"{step['step_id']}. {step['instruction']}" for step in procedure_steps if step["instruction"]
    )
    plan["observation_text"] = _first_text(plan, "observation_text") or "\n".join(
        f"{step['step_id']}. {step['observation']}" for step in procedure_steps if step["observation"]
    )
    plan.setdefault("description", "")
    plan.setdefault("subject", "")
    plan.setdefault("educational_goal", "")
    return plan


def _parse_json_response(response: str) -> dict[str, Any]:
    clean = response.strip()
    clean = re.sub(r"^```(?:json)?\s*", "", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\s*```$", "", clean).strip()
    try:
        parsed = json.loads(clean)
    except json.JSONDecodeError as error:
        print(f"[rag] Qwen returned invalid JSON: {error}", flush=True)
        print(f"[rag] response prefix: {response[:500]}", flush=True)
        raise ValueError("Qwen fallback did not return valid JSON") from error
    if not isinstance(parsed, dict):
        raise ValueError("Qwen fallback JSON must be an object")
    return parsed


def generate_procedure_with_llm(query: str) -> dict[str, Any]:
    """Generate and normalize a structured plan after a local RAG miss."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = "Qwen/Qwen2.5-3B-Instruct"
    tokenizer = None
    model = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16, device_map="auto"
        )
        system = """Generate ONLY valid JSON for an educational experiment video plan.
Required top-level keys: experiment_name, subject, educational_goal, procedure_text,
observation_text, clips. clips must contain 3-6 objects. Each clip must preserve
procedure_text, observation_text, state_before, action, state_after, reference_policy,
reference_clip, handoff_policy, handoff_entity, duration_seconds, image_prompt,
motion_prompt, negative_prompt. Use null or empty strings when a field is not applicable.
Do not use markdown fences."""
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Create a plan for: {query}"},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            output = model.generate(
                **inputs, max_new_tokens=1400, temperature=0.3, do_sample=True
            )
        response = tokenizer.decode(
            output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
        )
        return normalize_experiment_plan(_parse_json_response(response), query)
    except (OSError, RuntimeError, ValueError, ImportError) as error:
        print(f"[rag] Qwen fallback failed: {error}", flush=True)
        raise
    finally:
        if model is not None:
            del model
        if tokenizer is not None:
            del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def store(experiment: dict[str, Any]) -> Path:
    normalized = normalize_experiment_plan(experiment)
    name = normalized["name"]
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "unknown"
    path = KNOWLEDGE_DIR / f"{slug}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(normalized, indent=2), encoding="utf-8")
    print(f"[rag] stored new procedure: {path}")
    return path


def lookup_or_generate(query: str) -> dict[str, Any]:
    """Use local RAG first; invoke Qwen only when no record matches."""
    result = lookup(query)
    if result is not None:
        normalized = normalize_experiment_plan(result, query)
        print(f"[rag] cache hit: {normalized['experiment_name']}")
        return normalized
    print(f"[rag] miss for '{query}' - generating with Qwen")
    generated = normalize_experiment_plan(generate_procedure_with_llm(query), query)
    store(generated)
    return generated


if __name__ == "__main__":
    import sys

    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "newton's cradle"
    print(json.dumps(lookup_or_generate(query), indent=2))
