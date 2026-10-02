"""Adapt the pipeline JSON-lines generator protocol to Wan's Kaggle runner.

This wrapper is intended for Kaggle's two-GPU T4 session. Wan's streamed INT8
runner is a one-video process, so each pipeline clip starts a fresh child
process and repeats model setup/conversion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path


def duration_to_frames(seconds: float, fps: int = 16) -> int:
    """Return the smallest Wan-compatible 4n+1 frame count covering duration."""
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("clip duration must be a finite number greater than zero")
    return max(17, 4 * math.ceil((seconds * fps - 1) / 4) + 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wan-repo-dir", type=Path, required=True)
    parser.add_argument("--model-preset", choices=["i2v-a14b"], default="i2v-a14b")
    parser.add_argument("--max-memory-gib", type=float, default=8.0)
    parser.add_argument("--gpu0-memory-gib", type=float)
    parser.add_argument("--gpu1-memory-gib", type=float)
    parser.add_argument("--daemon", action="store_true")
    # Arguments emitted by the pipeline's regular command mode.
    parser.add_argument("--input-image")
    parser.add_argument("--output-video")
    parser.add_argument("--prompt")
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--clip-duration", type=float, default=1.0)
    parser.add_argument("--clip-name", default="")
    return parser.parse_args()


def run_clip(args: argparse.Namespace, task: dict[str, object]) -> None:
    repo = args.wan_repo_dir.resolve()
    runner = repo / "quantization" / "kaggle_generate_video.py"
    image = Path(str(task.get("input_image", args.input_image or ""))).resolve()
    output = Path(str(task.get("output_video", args.output_video or ""))).resolve()
    prompt = str(task.get("prompt", args.prompt or ""))
    negative = str(task.get("negative_prompt", args.negative_prompt or ""))
    duration = float(task.get("clip_duration", args.clip_duration))
    clip_name = str(task.get("clip_name", args.clip_name))

    if not runner.is_file():
        raise FileNotFoundError(f"Wan Kaggle runner not found: {runner}")
    if not image.is_file():
        raise FileNotFoundError(f"Input image not found: {image}")
    if not prompt.strip():
        raise ValueError("The clip prompt is empty")
    if not str(task.get("output_video", args.output_video or "")).strip():
        raise ValueError("No output video path was supplied")

    seed_material = clip_name or str(output)
    seed = int.from_bytes(hashlib.sha256(seed_material.encode("utf-8")).digest()[:4], "big")
    command = [
        sys.executable, str(runner),
        "--image", str(image),
        "--prompt", prompt,
        "--negative-prompt", negative,
        "--frames", str(duration_to_frames(duration)),
        "--steps", "4", "--lightning",
        "--max-memory-gib", str(args.max_memory_gib),
        "--seed", str(seed),
        "--output", str(output),
    ]
    if args.gpu0_memory_gib is not None:
        command.extend(["--gpu0-memory-gib", str(args.gpu0_memory_gib)])
    if args.gpu1_memory_gib is not None:
        command.extend(["--gpu1-memory-gib", str(args.gpu1_memory_gib)])
    output.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"Starting {clip_name or output.name}: {duration:g}s -> "
        f"{duration_to_frames(duration)} frames; model setup runs per clip.",
        file=sys.stderr,
        flush=True,
    )
    # Reserve wrapper stdout for READY/JSON lines consumed by LocalVideoGenerator.
    subprocess.run(command, cwd=repo, check=True, stdout=sys.stderr, stderr=sys.stderr)
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError(f"Wan completed without a non-empty video: {output}")


def daemon(args: argparse.Namespace) -> int:
    print("READY", flush=True)
    for line in sys.stdin:
        try:
            task = json.loads(line)
            if task.get("action") == "exit":
                return 0
            run_clip(args, task)
            print(json.dumps({"status": "success"}), flush=True)
        except Exception as error:
            print(json.dumps({"status": "error", "error": str(error)}), flush=True)
    return 0


def main() -> int:
    args = parse_args()
    if args.daemon:
        return daemon(args)
    run_clip(args, {
        "input_image": args.input_image,
        "output_video": args.output_video,
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "clip_duration": args.clip_duration,
        "clip_name": args.clip_name,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
