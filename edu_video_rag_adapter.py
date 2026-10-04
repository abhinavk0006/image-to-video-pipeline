"""Bridge Edu-video-gen-dataset retrieval bundles into pipeline knowledge JSON."""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_") or "clip"


def retrieve_pipeline_experiment(
    query: str,
    rag_repo_dir: str | Path,
    *,
    top_k: int = 1,
    clip_duration_seconds: float = 1.0,
    max_clips: int | None = None,
) -> dict[str, Any]:
    """Run the fork's TF-IDF prompt-bundle builder and adapt its best result.

    The caller supplies the starting image to the video pipeline. Image URLs in
    RAG records are references only and are intentionally not fetched here.
    """
    repo = Path(rag_repo_dir).expanduser().resolve()
    builder = repo / "scripts" / "build_rag_prompt_bundle.py"
    if not builder.is_file():
        raise FileNotFoundError(f"RAG prompt builder not found: {builder}")
    if top_k < 1:
        raise ValueError("RAG top_k must be at least 1")
    if clip_duration_seconds <= 0:
        raise ValueError("RAG clip duration must be greater than zero")
    if max_clips is not None and max_clips < 1:
        raise ValueError("max_clips must be at least 1 when specified")

    with tempfile.TemporaryDirectory(prefix="edu_video_rag_") as temp_dir:
        output = Path(temp_dir) / "prompt_bundle.json"
        command = [
            sys.executable,
            str(builder),
            query,
            "--top-k",
            str(top_k),
            "--output",
            str(output),
        ]
        result = subprocess.run(
            command,
            cwd=repo,
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode:
            detail = result.stderr.strip() or result.stdout.strip()
            raise RuntimeError(f"Edu-video RAG retrieval failed: {detail}")
        bundle = json.loads(output.read_text(encoding="utf-8"))

    retrieved = bundle.get("retrieved_experiments") or []
    if not retrieved:
        raise ValueError(f"Edu-video RAG returned no experiments for query: {query}")
    selected = retrieved[0]
    score = float(selected.get("retrieval_score", 0.0))
    if score <= 0:
        raise ValueError(
            f"No positive RAG match for {query!r}; refine the query or choose a knowledge JSON."
        )

    experiment_id = selected.get("experiment_id")
    prompt_inputs = [
        prompt
        for prompt in bundle.get("prompt_inputs", [])
        if prompt.get("experiment_id") == experiment_id and prompt.get("prompt")
    ]
    if not prompt_inputs:
        raise ValueError(f"RAG match {experiment_id!r} contains no usable step prompts")
    if max_clips is not None:
        prompt_inputs = prompt_inputs[:max_clips]

    scene_by_id = {
        scene.get("scene_id"): scene
        for scene in selected.get("scenes", [])
        if scene.get("scene_id")
    }
    clips = []
    for index, prompt in enumerate(prompt_inputs, start=1):
        scene = scene_by_id.get(prompt.get("scene_id"), {})
        step_id = str(prompt.get("step_id", index))
        clip_name = f"clip_{index}_{_slug(step_id)}"
        clips.append(
            {
                "name": clip_name,
                "scene": {
                    "scene": str(scene.get("description", selected.get("title", ""))),
                    "camera": "Stable educational close view",
                    "lighting": "Clear, consistent laboratory lighting",
                    "objects": [],
                    "background": "",
                    "metadata": {"scene_id": prompt.get("scene_id")},
                },
                "prompt_bundle": {
                    "motion_prompt": prompt["prompt"],
                    "negative_prompt": prompt.get("negative_prompt", ""),
                    "clip_duration_seconds": float(
                        prompt.get("duration_seconds") or clip_duration_seconds
                    ),
                },
                "metadata": {
                    "experiment_id": experiment_id,
                    "step_id": step_id,
                    "source_ids": prompt.get("source_ids", selected.get("source_ids", [])),
                    "retrieval_score": score,
                    "source_status": selected.get("status"),
                },
            }
        )

    return {
        "name": selected.get("title") or experiment_id or query,
        "description": selected.get("educational_goal", ""),
        "subject": selected.get("subject", ""),
        "educational_goal": selected.get("educational_goal", ""),
        "knowledge_source": "Edu-video-gen-dataset TF-IDF RAG",
        "metadata": {
            "query": query,
            "experiment_id": experiment_id,
            "retrieval_score": score,
            "source_ids": selected.get("source_ids", []),
            "source_status": selected.get("status"),
        },
        "clips": clips,
    }
