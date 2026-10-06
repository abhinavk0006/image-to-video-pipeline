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


DEFAULT_MAX_AREA = 480 * 832


def is_cuda_oom_error(message: str) -> bool:
    """Return whether a child runner failure is a CUDA out-of-memory failure."""
    normalized = message.lower()
    return (
        "out of memory" in normalized
        or "cuda_error_out_of_memory" in normalized
        or "cublas_status_alloc_failed" in normalized
    )


def configured_areas(args: argparse.Namespace) -> tuple[int, ...]:
    """Return the requested area followed by configured OOM fallbacks."""
    areas = (args.max_area or DEFAULT_MAX_AREA, *args.max_area_fallback)
    if any(area <= 0 for area in areas):
        raise ValueError("configured max areas must be greater than zero")
    if len(set(areas)) != len(areas):
        raise ValueError("configured max areas must be unique")
    if any(areas[index + 1] >= areas[index] for index in range(len(areas) - 1)):
        raise ValueError("fallback max areas must strictly decrease")
    return areas


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
    parser.add_argument("--max-area", type=int)
    parser.add_argument(
        "--max-area-fallback",
        type=int,
        action="append",
        default=[],
        help="Lower pixel area to try after CUDA OOM. Repeat in descending order.",
    )
    parser.add_argument("--memory-telemetry", action="store_true")
    parser.add_argument(
        "--lightning",
        action="store_true",
        help="Use the Wan 2.2 Lightning four-step path (the Kaggle adapter's only supported mode).",
    )
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
    if args.memory_telemetry:
        command.append("--memory-telemetry")
    output.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"Starting {clip_name or output.name}: {duration:g}s -> "
        f"{duration_to_frames(duration)} frames; model setup runs per clip.",
        file=sys.stderr,
        flush=True,
    )
    # Reserve wrapper stdout for READY/JSON lines consumed by LocalVideoGenerator.
    requested_area = task.get("max_area", args.max_area)
    area_args = argparse.Namespace(
        max_area=int(requested_area) if requested_area is not None else None,
        max_area_fallback=args.max_area_fallback,
    )
    areas = configured_areas(area_args)
    for attempt, area in enumerate(areas):
        attempt_command = [*command, "--max-area", str(area)]
        try:
            subprocess.run(
                attempt_command,
                cwd=repo,
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as error:
            details = (error.stderr or error.stdout or "").strip()
            if details:
                print(details, file=sys.stderr, flush=True)
            if is_cuda_oom_error(details) and attempt + 1 < len(areas):
                print(
                    f"CUDA OOM at area {area}; retrying at area {areas[attempt + 1]}.",
                    file=sys.stderr,
                    flush=True,
                )
                output.unlink(missing_ok=True)
                continue
            raise RuntimeError(
                f"Wan generation failed at selected area {area} "
                f"(exit status {error.returncode}): {details}"
            ) from error
        if not output.is_file() or output.stat().st_size == 0:
            raise RuntimeError(
                f"Wan completed without a non-empty video at selected area {area}: {output}"
            )
        print(f"Selected Wan area {area} for {clip_name or output.name}.", file=sys.stderr)
        return
    raise AssertionError("generation area fallback loop was empty")


def build_wan_daemon_command(args: argparse.Namespace) -> list[str]:
    """Build the long-lived Wan runner command used by wrapper daemon mode."""
    repo = args.wan_repo_dir.resolve()
    runner = repo / "quantization" / "kaggle_generate_video.py"
    command = [
        sys.executable, str(runner), "--daemon",
        "--steps", "4", "--lightning",
        "--max-memory-gib", str(args.max_memory_gib),
    ]
    if args.gpu0_memory_gib is not None:
        command.extend(["--gpu0-memory-gib", str(args.gpu0_memory_gib)])
    if args.gpu1_memory_gib is not None:
        command.extend(["--gpu1-memory-gib", str(args.gpu1_memory_gib)])
    if args.max_area is not None:
        command.extend(["--max-area", str(args.max_area)])
    if args.memory_telemetry:
        command.append("--memory-telemetry")
    return command


def daemon(args: argparse.Namespace) -> int:
    command = build_wan_daemon_command(args)
    try:
        process = subprocess.Popen(
            command,
            cwd=args.wan_repo_dir.resolve(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            stderr=sys.stderr,
        )
        ready_line = process.stdout.readline()
        if ready_line.strip() != "READY":
            raise RuntimeError(f"Wan daemon failed to start: {ready_line.strip()}")
        print("READY", flush=True)
        for line in sys.stdin:
            task = json.loads(line)
            if task.get("action") == "exit":
                process.stdin.write(json.dumps({"action": "exit"}) + "\n")
                process.stdin.flush()
                process.wait(timeout=30)
                return process.returncode or 0
            process.stdin.write(json.dumps(task) + "\n")
            process.stdin.flush()
            response = process.stdout.readline()
            if not response:
                raise RuntimeError("Wan daemon terminated during clip generation")
            print(response.strip(), flush=True)
        return 0
    except Exception as error:
        print(json.dumps({"status": "error", "error": str(error)}), flush=True)
        return 1
    finally:
        if "process" in locals() and process.poll() is None:
            process.kill()


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
