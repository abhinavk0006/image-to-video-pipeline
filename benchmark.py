"""Reproducible sequence benchmarks with explicit human scientific review.

Run generation cases only on the target GPU machine. `collect` can inspect an
existing run without loading a model. Pixel similarity never certifies science.
"""
from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
import uuid

from frame_extractor import FrameExtractor

SCIENCE_CHECKS = (
    "source_verified", "apparatus_identity_preserved", "procedure_order_correct",
    "observation_on_correct_target", "observation_scientifically_correct", "pacing_readable",
    "liquid_appearance_matches_source", "liquid_quantity_and_continuity_correct",
    "tool_contact_matches_procedure",
)


def read_json(path, default=None):
    return json.loads(Path(path).read_text(encoding="utf-8")) if Path(path).is_file() else default


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2), encoding="utf-8")


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def probe_video(path):
    import imageio_ffmpeg
    reader = imageio_ffmpeg.read_frames(str(path), pix_fmt="rgb24")
    try:
        meta = next(reader)
        count = sum(1 for _ in reader)
        fps = float(meta["fps"])
        if count < 1 or not math.isfinite(fps) or fps <= 0:
            raise ValueError("Video must contain decoded frames and a valid FPS")
        return {"frames": count, "fps": fps, "width": meta["size"][0],
            "height": meta["size"][1], "seconds": count / fps}
    finally:
        reader.close()


def review_status(checks):
    if any(value is not None and not isinstance(value, bool) for value in checks.values()):
        raise ValueError("Scientific review checks must be true, false or null")
    if any(value is False for value in checks.values()):
        return "failed"
    return "passed" if checks and all(value is True for value in checks.values()) else "pending"


def collect_run(knowledge_file, experiment_dir, report_dir, *, wall_seconds=None,
                generation_status="unknown", log_path=None):
    """Inspect clips in declared sequence order and create a reviewable report."""
    started = time.perf_counter()
    knowledge_file, experiment_dir, report_dir = map(Path, (knowledge_file, experiment_dir, report_dir))
    report_dir.mkdir(parents=True, exist_ok=True)
    knowledge = read_json(knowledge_file)
    from knowledge_loader import load_experiment_from_json
    from material_state import prepare_state_tracks
    resolved_experiment = load_experiment_from_json(str(knowledge_file))
    prepare_state_tracks(resolved_experiment)
    resolved_clips = {clip.name: clip for clip in resolved_experiment.clips}
    review_path = report_dir / "scientific_review.json"
    review = read_json(review_path, {"reviewer": "", "source_reference": "", "clips": {}})
    knowledge_hash = file_sha256(knowledge_file)
    records, seen = [], set()
    for index, clip in enumerate(knowledge.get("clips", [])):
        name = clip.get("name") or f"clip_{index + 1}"
        if name in seen or Path(name).name != name:
            raise ValueError("Benchmark clip names must be unique plain file names")
        resolved = resolved_clips[name]
        dependency = resolved.reference_clip
        dependency_valid = not dependency or dependency in seen
        seen.add(name)
        video = experiment_dir / f"{name}.mp4"
        required = [*SCIENCE_CHECKS, *(["handoff_state_matches"] if dependency or clip.get("continuity_required") else [])]
        entry = review["clips"].setdefault(name, {"checks": {}, "notes": ""})
        evidence = {"knowledge_sha256": knowledge_hash,
                    "video_sha256": file_sha256(video) if video.is_file() else None}
        if entry.get("evidence") != evidence and any(value is not None for value in entry["checks"].values()):
            entry.setdefault("previous_reviews", []).append({
                "checks": entry["checks"], "notes": entry.get("notes", ""),
                "evidence": entry.get("evidence"),
                "invalidated_reason": "Generated video/source changed or the old review had no evidence hashes",
            })
            entry["checks"], entry["notes"] = {}, ""
        entry["evidence"] = evidence
        for key in required:
            entry["checks"].setdefault(key, None)
        science = review_status({key: entry["checks"][key] for key in required})
        row = {"clip": name, "reference_clip": dependency, "dependency_order_valid": dependency_valid,
            "reference_provenance": read_json(experiment_dir / name / "reference_provenance.json", {}),
            "expected_liquid_state": resolved.liquid_state,
            "technical_status": "failed", "scientific_status": science,
            "generation": read_json(video.with_suffix(".generation.json"), {}),
            "worker": read_json(video.with_suffix(".metrics.json"), {})}
        try:
            info = probe_video(video)
            handoffs = {}
            for mode in ("mid", "near_end", "final"):
                image = report_dir / f"{name}_{mode}.png"
                FrameExtractor().extract_handoff_frame(video, image, mode=mode)
                handoffs[mode] = read_json(Path(str(image) + ".json"))
            expected = {"mid": (info["frames"] - 1) // 2,
                "near_end": max(0, info["frames"] - 1 - round(.5 * info["fps"])),
                "final": info["frames"] - 1}
            valid = dependency_valid and all(handoffs[mode]["frame_index"] == value for mode, value in expected.items())
            requested = clip.get("duration_seconds", clip.get("prompt_bundle", {}).get("clip_duration_seconds"))
            if requested is not None:
                expected_frames = max(17, 4 * math.ceil((float(requested) * info["fps"] - 1) / 4) + 1)
                row["expected_wan_frames"] = expected_frames
                valid = valid and info["frames"] == expected_frames
            row.update(video=info, handoffs=handoffs, technical_status="passed" if valid else "failed")
        except Exception as error:
            row["technical_error"] = str(error)
        records.append(row)
    write_json(review_path, review)
    durations = [row["video"]["seconds"] for row in records if "video" in row]
    pending = sum(row["scientific_status"] == "pending" for row in records)
    accepted = sum(row.get("video", {}).get("seconds", 0) for row in records
                   if row["technical_status"] == row["scientific_status"] == "passed")
    log = Path(log_path).read_text(encoding="utf-8", errors="replace") if log_path and Path(log_path).exists() else ""
    if wall_seconds is not None and (not math.isfinite(wall_seconds) or wall_seconds <= 0):
        raise ValueError("Pipeline wall time must be a finite positive number")
    report = {"schema_version": 1, "experiment": knowledge.get("name"),
        "generation_status": generation_status, "knowledge_sha256": hashlib.sha256(knowledge_file.read_bytes()).hexdigest(),
        "pipeline_wall_seconds": wall_seconds, "generated_video_seconds": sum(durations),
        "longest_generated_clip_seconds": max(durations, default=0),
        "expected_clips": len(knowledge.get("clips", [])),
        "decodable_clips": len(durations), "scientific_reviews_pending": pending,
        "accepted_generated_seconds": accepted if not pending else None,
        "accepted_seconds_per_wall_second": accepted / wall_seconds if wall_seconds and not pending else None,
        "generated_seconds_per_wall_second": sum(durations) / wall_seconds if wall_seconds else None,
        "cuda_oom_detected": True if any(row["worker"].get("status") == "cuda_oom" for row in records)
            or re.search(r"cuda[^\n]*out of memory|cuda_oom|cuda_error_out_of_memory", log, re.I)
            else False if log_path else None,
        "clips": records,
        "measurement_notes": ["Playback slowdown and text cards do not count as generated video seconds.",
            "Scientific checks are human review, not an automatic accuracy score.",
            "Worker phase memory peaks are cumulative since each request began; nested expert-load times must not be added to denoising time."],
        "analysis_wall_seconds": time.perf_counter() - started}
    peaks, stages = {}, {}
    for row in records:
        for gpu in row["worker"].get("cuda", []):
            device = str(gpu["device"])
            peaks[device] = max(peaks.get(device, 0), gpu["peak_allocated_bytes"] / 2**30)
        for phase in row["worker"].get("phases", []):
            stages[phase["stage"]] = stages.get(phase["stage"], 0) + phase["wall_seconds"]
    report.update(peak_allocated_gib_by_device=peaks, stage_seconds=stages,
        expert_loading_seconds=sum(load["wall_seconds"] for row in records for load in row["worker"].get("expert_loads", [])),
        incomplete_worker_stages=sorted({stage for row in records for stage in row["worker"].get("incomplete_stages", [])}))
    final = experiment_dir / "final_video.mp4"
    try:
        report["stitched_sequence"] = probe_video(final)
    except Exception as error:
        report["stitched_sequence_error"] = str(error)
    sequence_duration_matches = "stitched_sequence" in report and abs(report["stitched_sequence"]["seconds"] - sum(durations)) <= max(.25, len(records) / 16)
    report["technical_sequence_complete"] = bool(records) and all(row["technical_status"] == "passed" for row in records) and sequence_duration_matches
    report["analysis_wall_seconds"] = time.perf_counter() - started
    write_json(report_dir / "benchmark.json", report)
    return report


def environment_record(pipeline, wan):
    record = {"python": sys.version, "platform": platform.platform(), "repositories": {}}
    for repo in (pipeline, wan):
        def git(*args):
            result = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
            return result.stdout.strip() if result.returncode == 0 else None
        tracked = git("ls-files", "*.py") or ""
        hashes = {name: hashlib.sha256((repo / name).read_bytes()).hexdigest()
                  for name in tracked.splitlines() if (repo / name).is_file()}
        record["repositories"][repo.name] = {"commit": git("rev-parse", "HEAD"),
            "dirty": bool(git("status", "--porcelain")), "python_source_sha256": hashes}
    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10)
        record["gpu_inventory"] = gpu.stdout.strip() if gpu.returncode == 0 else "unavailable"
    except (OSError, subprocess.TimeoutExpired):
        record["gpu_inventory"] = "unavailable"
    return record


def summarize(run_root):
    """Refresh aggregate results after individual scientific reviews change."""
    run_root = Path(run_root)
    rows = []
    for path in sorted(run_root.glob("*/case.json")):
        case = read_json(path)
        report = read_json(path.parent / "report/benchmark.json")
        if report is None:
            continue
        rows.append({"case": path.parent.name, "seconds": case["settings"]["seconds"],
            "max_area": case["settings"]["max_area"], "exit_code": case.get("exit_code"),
            "gpu0_peak_allocated_gib": report["peak_allocated_gib_by_device"].get("0"),
            "gpu1_peak_allocated_gib": report["peak_allocated_gib_by_device"].get("1"),
            "expert_loading_seconds": report["expert_loading_seconds"],
            "denoise_including_loading_seconds": report["stage_seconds"].get("denoise_including_expert_loads"),
            "vae_decode_seconds": report["stage_seconds"].get("vae_decode"),
            **{key: report[key] for key in ("pipeline_wall_seconds", "generated_video_seconds", "cuda_oom_detected", "technical_sequence_complete", "scientific_reviews_pending", "accepted_seconds_per_wall_second")}})
    write_json(run_root / "summary.json", rows)
    if rows:
        with (run_root / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return rows


def run_matrix(args):
    pipeline = Path(__file__).resolve().parent
    wan = Path(args.wan_repo_dir).resolve()
    matrix = read_json(args.matrix)
    cases = matrix["cases"]
    if args.cases:
        requested = set(args.cases)
        if requested - {case["name"] for case in cases}:
            raise ValueError("Unknown benchmark case")
        cases = [case for case in cases if case["name"] in requested]
    elif not args.all_cases:
        cases = cases[:1]  # A full GPU sweep must be explicitly selected.
    knowledge = read_json(args.knowledge_file)
    run_root = Path(args.output_dir).resolve() / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8])
    run_root.mkdir(parents=True)
    write_json(run_root / "environment.json", environment_record(pipeline, wan))
    for case in cases:
        name = case["name"]
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", name):
            raise ValueError("Case names must contain letters, digits, hyphens or underscores")
        seconds, area = float(case["seconds"]), int(case["max_area"])
        if not math.isfinite(seconds) or seconds <= 0 or area <= 0:
            raise ValueError("Each benchmark case needs a positive duration and area")
        directory = run_root / name
        directory.mkdir()
        fixture = copy.deepcopy(knowledge)
        for clip in fixture["clips"]:
            clip["duration_seconds"] = seconds
        fixture_file = directory / "experiment.json"
        write_json(fixture_file, fixture)
        command = [sys.executable, str(pipeline / "main.py"), "--knowledge-file", str(fixture_file),
            "--output-dir", str(directory / "videos"), "--experiment-name", "sequence",
            "--generator-script", str(pipeline / "kaggle_wan_wrapper.py"), "--generator-executable", sys.executable,
            "--generator-working-dir", str(pipeline), "--wan-repo-dir", str(wan), "--wan-model-preset", "i2v-a14b",
            "--continuity-mode", "chain", "--wan-max-area", str(area), "--wan-memory-telemetry",
            "--generator-arg=--profile-stages", "--generator-arg=--gpu0-memory-gib=5", "--generator-arg=--gpu1-memory-gib=11"]
        if args.initial_image:
            command += ["--initial-image", str(Path(args.initial_image).resolve())]
        if args.expert_source_dir:
            command.append(f"--generator-arg=--expert-source-dir={Path(args.expert_source_dir).resolve()}")
        # No resolution fallback in controlled comparisons: an OOM is a result.
        case_record = {"settings": case, "command": command,
            "worker_lifecycle": "fresh process per case, persistent across that case's clips",
            "seed_policy": "SHA256 of clip name; unchanged across cases",
            "filesystem_download_cache": "not cleared", "fallbacks_enabled": False}
        write_json(directory / "case.json", case_record)
        log_path = directory / "generation.log"
        print(f"Benchmark {name}; logging to {log_path}", flush=True)
        started = time.perf_counter()
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=pipeline, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, errors="replace")
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            code = process.wait()
        elapsed = time.perf_counter() - started
        case_record.update(exit_code=code, pipeline_wall_seconds=elapsed)
        write_json(directory / "case.json", case_record)
        report = collect_run(fixture_file, directory / "videos/sequence", directory / "report",
            wall_seconds=elapsed, generation_status="success" if code == 0 else "failed", log_path=log_path)
        summarize(run_root)
    print(f"Benchmark reports: {run_root}")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    summary = sub.add_parser("summarize", help="Refresh JSON/CSV summaries after scientific review")
    summary.add_argument("--run-dir", required=True)
    collect = sub.add_parser("collect", help="Inspect an existing sequence without GPU generation")
    collect.add_argument("--knowledge-file", required=True)
    collect.add_argument("--experiment-dir", required=True)
    collect.add_argument("--report-dir", required=True)
    collect.add_argument("--pipeline-wall-seconds", type=float)
    run = sub.add_parser("run", help="Generate fixed configurations on the target GPU machine")
    run.add_argument("--knowledge-file", required=True)
    run.add_argument("--wan-repo-dir", required=True)
    run.add_argument("--output-dir", required=True)
    run.add_argument("--initial-image")
    run.add_argument("--expert-source-dir")
    run.add_argument("--matrix", default=str(Path(__file__).with_name("benchmark_matrix.json")))
    run.add_argument("--cases", nargs="+")
    run.add_argument("--all-cases", action="store_true")
    args = parser.parse_args(argv)
    if args.operation == "summarize":
        print(json.dumps(summarize(args.run_dir), indent=2))
        return 0
    if args.operation == "run":
        return run_matrix(args)
    report = collect_run(args.knowledge_file, args.experiment_dir, args.report_dir,
                         wall_seconds=args.pipeline_wall_seconds)
    print(json.dumps({key: value for key, value in report.items() if key != "clips"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
