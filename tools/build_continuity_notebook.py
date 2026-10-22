"""Build a Kaggle notebook pinned to complete source repository snapshots."""
import json
from pathlib import Path
import subprocess
import re

import argparse
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--wan-repo", type=Path)
parser.add_argument("--rag-repo", type=Path)
parser.add_argument("--output-dir", type=Path)
args = parser.parse_args()
REPO = Path(__file__).resolve().parents[1]
PROJECTS = REPO.parent
OUTPUT = args.output_dir or REPO / "artifacts"
OUTPUT.mkdir(parents=True, exist_ok=True)
SOURCES = {"image-to-video-pipeline": REPO,
           "Wan2.2": args.wan_repo or PROJECTS / "Wan2.2",
           "Edu-video-gen-dataset": args.rag_repo or PROJECTS / "Edu-video-gen-dataset"}
# Pin complete repository snapshots; never overlay partial old sources onto a new dataset.
BASES = {name: subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=directory,
                                    text=True).strip() for name, directory in SOURCES.items()}
manifest = {"base_commits": BASES, "source_policy": "complete pinned Git snapshots"}
cells = []


def md(source):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": source.strip() + "\n"})


def code(source):
    # subprocess cwd only affects the child. Each notebook cell importing local
    # modules must establish the kernel's path itself, including on partial reruns.
    modules = sorted(set(re.findall(
        r"^(?:from|import) (knowledge_loader|models|material_state|material_plan|science_cards|frame_extractor|stitcher|edu_video_rag_adapter|source_quantities)\b",
        source, re.MULTILINE)))
    if modules:
        bootstrap = '''import sys, importlib
from pathlib import Path
pipeline = Path("/kaggle/working/image-to-video-pipeline").resolve()
required_pipeline_files = REQUIRED_FILES
missing_pipeline_files = [name for name in required_pipeline_files if not (pipeline / name).is_file()]
if missing_pipeline_files:
    raise FileNotFoundError(f"Pipeline sources missing at {pipeline}: {missing_pipeline_files}. Restore the notebook's source setup in this session; existing videos do not need regeneration.")
sys.path[:] = [str(pipeline)] + [entry for entry in sys.path if entry != str(pipeline)]
importlib.invalidate_caches()
'''.replace("REQUIRED_FILES", repr([name + ".py" for name in modules]))
        source = bootstrap + source
    cells.append({"cell_type": "code", "metadata": {}, "source": source.strip() + "\n", "outputs": [], "execution_count": None})


md('''# Persistent Wan worker: source quantities, continuous state and styled cards

Use Kaggle **Internet + two T4 GPUs**. All three repositories are pinned to complete
Git snapshots. This notebook consumes the selected dataset protocol rather than a
hard-coded four-shot demonstration. Source quantities, their roles and page citations
reach the plan and cards. Unknown doses remain unknown.

The default chloride confirmation starts with a **prepared 1 mL test aliquot**
(source step 1), then renders source steps 2–4. It does not include dissolving dry
salt. Supply a matching starting image; the old dry-salt image is unsuitable here.
Other protocols require their own reviewed sample mapping and reference images.

The default safe_2s profile retains 33 frames. Longer shots remain experimental.
Playback slowdown adds no generated frames and cannot repair incorrect chemistry.
Four-step Lightning uses CFG=1, so constraints are carried in the positive prompt.''')
code('''import base64, gc, hashlib, io, json, os, subprocess, sys, time, zipfile
from pathlib import Path
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
ROOT = Path("/kaggle/working")
EXPERIMENT = "chloride"  # "chloride", "dataset", or the separate "litmus" regression fixture
DATASET_EXPERIMENT_ID = "chem_12_301"
DATASET_PROTOCOL = "chloride_silver"
PROTOCOL_START_STEP = 2  # chloride step 1 prepares the source's 1 mL aliquot
INITIAL_IMAGE_NAME = "chloride_1ml_aliquot.png"  # must match the selected starting state
REFERENCE_IMAGES = {}  # other clip names -> mounted image paths, for separate samples
# Explicitly reviewed mapping for this same-aliquot chloride confirmation only.
# For another protocol, replace this mapping; generic sample_id='reaction' is not proof of continuity.
SAMPLE_TRACKS = {"clip_2_2": "chloride_aliquot", "clip_3_3": "chloride_aliquot", "clip_4_4": "chloride_aliquot"}
USE_VAE_CPU_STREAMING = False  # opt-in memory comparison; does not lower attention activation memory
STREAM_EXPERT_TO_GPU = False  # opt-in conversion path; validate separately
CPU_EXPERT_CACHE_GIB = 14  # high expert only; host headroom guard may evict/skip it
EXPERT_CACHE_DIR = None  # optional converted disk cache; two experts need about 27 GiB
PROMPT_CACHE_DIR = ROOT / "wan_prompt_cache"
PROFILE = "safe_2s"     # "safe_2s", "fast_preview", or "experimental_3s"
SAMPLING_MODE = "lightning"  # optional "base": slower 20-step CFG comparison, no Lightning LoRA
BASE_STEPS = 20
PROFILES = {
    "safe_2s": {"seconds": 2.0, "area": 345600, "fallbacks": [230400, 172800]},
    "fast_preview": {"seconds": 2.0, "area": 230400, "fallbacks": [172800]},
    "experimental_3s": {"seconds": 3.0, "area": 230400, "fallbacks": [172800]},
}
SETTINGS = PROFILES[PROFILE]
PLAYBACK_RATE = 0.5       # 33 frames at 16 FPS -> about 4.125 presented seconds; no extra generated frames
EXPERT_SOURCE_DIR = None  # optional /kaggle/input/... folder with the two Comfy FP8 expert files
PROFILE_STAGES = True     # synchronizes CUDA phase boundaries and saves timing/memory sidecars
MATERIAL_PLAN_MODE = "strict"  # requires documented doses and a cumulative vessel budget before GPU work
MATERIAL_REVIEW_MODE = "observe"  # "strict" pauses at each clip for hash-bound human review before chaining
SCIENCE_MODE = "observe"  # save evidence/review scaffolds; pending reviews are never accuracy scores
MATERIAL_OVERRIDES = {}   # clip name -> source-backed liquid_state fields; no guessed quantities
assert EXPERIMENT in {"chloride", "dataset", "litmus"} and 0 < PLAYBACK_RATE <= 1
OUT = ROOT / f"wan_continuous_state_{EXPERIMENT}_{PROFILE}_{SAMPLING_MODE}"
OUT.mkdir(parents=True, exist_ok=True)
print(SETTINGS, "playback rate:", PLAYBACK_RATE)''')
code('''manifest = ''' + repr(manifest) + '''
refs = {"Wan2.2": "feature/persistent-wan-worker", "image-to-video-pipeline": "feature/persistent-wan-worker", "Edu-video-gen-dataset": "main"}
for name, revision in manifest["base_commits"].items():
    path = ROOT / name
    if not path.exists():
        subprocess.run(["git", "clone", "--branch", refs[name], f"https://github.com/abhinavk0006/{name}.git", str(path)], check=True)
    current = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True).strip()
    if current != revision:
        dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=path, text=True).strip()
        if dirty:
            raise RuntimeError(f"{name} has edits at a different revision. Preserve that clone and use a fresh Kaggle session.")
        subprocess.run(["git", "fetch", "origin", refs[name]], cwd=path, check=True)
        subprocess.run(["git", "checkout", "--detach", revision], cwd=path, check=True)
pipeline = ROOT / "image-to-video-pipeline"
sys.path.insert(0, str(pipeline))
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(ROOT / "Edu-video-gen-dataset" / "requirements.txt"),
    "diffusers>=0.31,<0.33", "transformers>=4.49,<=4.51.3", "accelerate>=1.1.1", "easydict", "safetensors", "ftfy", "huggingface_hub", "imageio", "imageio-ffmpeg", "Pillow>=10.1", "matplotlib"], check=True)
(OUT / "source_manifest.json").write_text(json.dumps(manifest, indent=2))
print("Complete pinned source snapshots:", manifest["base_commits"])''')
md('''## Source quantities and material preflight
The selected source records distinguish aliquots, reagent doses, stocks, capacities,
conditions and endpoints. For chloride the initial aliquot is 1 mL, but acid,
AgNO3, NH4OH doses and vessel working capacity are unspecified. Supply missing facts
through `MATERIAL_OVERRIDES` in the configuration cell using the schema in
`MATERIAL_BUDGETS.md`. Do not assume equal doses or a universal drop volume.

The default `MATERIAL_PLAN_MODE="strict"` stops the generation cell before model
startup if the dose/volume budget is unresolved. The plan and styled cards can
still be inspected. `warn` permits an explicitly unvalidated visual trial; it
does not make unknown quantities scientifically valid. Existing videos stay saved.

`MATERIAL_REVIEW_MODE="strict"` additionally pauses before a liquid clip is accepted
or chained until its `.material-review.json` contains independent source-linked,
timestamped human review. `observe` records pending reviews and permits continuation;
an explicitly failed review always stops propagation. No automatic chemistry
verifier is supplied.
''')
md('''## Build the selected source plan
The same declared sample uses the actual final decoded frame of its preceding clip.
Fresh aliquots and changed apparatus require separate references. The notebook rejects
unsplit composite actions. Draft scientific requirements and visibility/evidence tools
remain available; they are not an automatic chemical-accuracy verifier.

Plan and cards can be inspected before supplying an image. The generation cell checks
that the matching initial reference is mounted. Inspect every reference and raw output.''')
code('''from edu_video_rag_adapter import retrieve_pipeline_experiment
from knowledge_loader import _parse_clip
from models import Experiment
from material_state import prepare_state_tracks
from material_plan import compile_material_plan

def unique_input(name):
    matches = list(Path("/kaggle/input").rglob(name))
    if len(matches) != 1:
        raise RuntimeError(f"Mount exactly one {name}; found {len(matches)}")
    return matches[0]

if EXPERIMENT == "litmus":
    fixture = json.loads((pipeline / "knowledge/ammonia_litmus_observation.json").read_text())
    INITIAL_IMAGE_NAME = "ammonia_litmus_ready.png"
    INITIAL_IMAGE = None
else:
    rag = ROOT / "Edu-video-gen-dataset"
    # Rebuild the portable TF-IDF artifacts with this kernel's sklearn version.
    subprocess.run([sys.executable, "scripts/build_rag_index.py"], cwd=rag, check=True)
    fixture = retrieve_pipeline_experiment(
        "Detect one cation and one anion in a salt" if EXPERIMENT == "chloride" else DATASET_EXPERIMENT_ID,
        rag, clip_duration_seconds=SETTINGS["seconds"],
        protocol_selections={DATASET_EXPERIMENT_ID: DATASET_PROTOCOL})
    full_protocol = list(fixture["clips"])
    assert 1 <= PROTOCOL_START_STEP <= len(full_protocol)
    fixture["metadata"]["omitted_preparation_steps"] = full_protocol[:PROTOCOL_START_STEP-1]
    fixture["clips"] = full_protocol[PROTOCOL_START_STEP-1:]
    INITIAL_IMAGE = None
    if EXPERIMENT == "chloride":
        assert (DATASET_EXPERIMENT_ID, DATASET_PROTOCOL, PROTOCOL_START_STEP) == ("chem_12_301", "chloride_silver", 2)
        # This is the volume of the prepared aliquot, never a reagent dose.
        fixture["clips"][0].setdefault("liquid_state", {}).update(before_ml=1.0)
        fixture["metadata"]["initial_sample_evidence"] = full_protocol[0]["metadata"]["source_evidence"]
        fixture["description"] = "Start with the source's prepared 1 mL test aliquot; review its identity, appearance and acidification endpoint."
        fixture["educational_goal"] = fixture["description"]
    fixture["name"] = f"{DATASET_EXPERIMENT_ID} — {DATASET_PROTOCOL}"
    fixture["conclusion_text"] = "Interpret the documented observation only after reviewing the complete procedure and generated evidence."

clips = fixture["clips"]
for item in clips:
    if item.get("requires_clip_split"):
        raise ValueError(f"{item['name']} combines operations. Split it into reviewed atomic shots before GPU work.")
    item["duration_seconds"] = SETTINGS["seconds"]
    # Discard legacy catalogue image URLs; source selection needs reviewed starting states.
    item["input_frame_path"] = REFERENCE_IMAGES.get(item["name"])
    item["state_track"] = SAMPLE_TRACKS.get(item["name"], "") if EXPERIMENT != "litmus" else ""
    if item is not clips[0] and not (item["state_track"] or item["input_frame_path"] or item.get("reference_clip")):
        raise ValueError(f"{item['name']}: explicitly map the sample track or provide its separate reference; do not infer continuity from adjacent steps.")
    item["liquid_state"] = {**item.get("liquid_state", {}), **MATERIAL_OVERRIDES.get(item["name"], {})}
    item.setdefault("procedure_text", item.get("action", ""))
    item.setdefault("observation_text", item.get("state_after", ""))
    print(item["name"], item["procedure_text"], item.get("metadata", {}).get("source_quantity_citation", ""), sep=" | ")
plan = Experiment(name=fixture["name"], clips=[_parse_clip(item, index) for index, item in enumerate(clips)])
prepare_state_tracks(plan)
material_report = compile_material_plan(plan)
(OUT / "material_plan.json").write_text(json.dumps(material_report, indent=2))
print("Material plan:", material_report["status"])
for row in material_report["clips"]:
    print(row["clip"], "before/after mL:", row["expected_before_ml"], row["expected_after_ml"], "issues:", row["issues"])
for item, resolved in zip(clips, plan.clips):
    item.update(reference_clip=resolved.reference_clip, reference_mode=resolved.reference_mode,
                continuity_required=resolved.continuity_required, handoff_policy=resolved.handoff_policy)
experiment_slug = "selected_source_protocol" if EXPERIMENT != "litmus" else "ammonia_litmus_observation"
fixture_path = OUT / "experiment.json"
fixture_path.write_text(json.dumps(fixture, indent=2))
print("Shots:", [item["name"] for item in clips])''')
code('''from PIL import Image
from science_cards import render_cards
cards_dir = OUT / "cards"
def cards(name, heading, body, kind):
    subject = "physics" if "physics" in fixture.get("subject", "").lower() else "chemistry"
    return render_cards(cards_dir, name, heading, body, subject=subject, kind=kind,
                        footer="One operation at a time • Observe the change")
title_cards = cards("00_title", fixture["name"], fixture["educational_goal"], "title")
procedure_cards, observation_cards = [], []
for index, item in enumerate(clips, 1):
    procedure_cards.append(cards(f"{index:02d}_procedure", f"Step {index} · Procedure", item["procedure_text"], "procedure"))
    observation_cards.append(cards(f"{index:02d}_observation", "What to observe", item["observation_text"], "observation"))
conclusion_cards = cards("99_conclusion", "Conclusion", fixture.get("conclusion_text", "Review the documented observation."), "conclusion")
from IPython.display import display
display(Image.open(procedure_cards[0][0]))
gc.collect()''')
md('''## Generate and inspect
The worker caches positive T5 contexts once. It remains persistent, but the streamed experts still need loading and conversion. Optional `EXPERT_SOURCE_DIR` reads mounted expert weights directly and avoids those repeated network transfers. Both filenames must match the original Comfy checkpoints.

Inspect raw clips and the three handoff images below. Check vessel count, target of colour change, action order, and whether the final state actually exists before chaining.''')
code('''INITIAL_IMAGE = unique_input(INITIAL_IMAGE_NAME)
clips[0]["input_frame_path"] = str(INITIAL_IMAGE)
fixture_path.write_text(json.dumps(fixture, indent=2))
command = [sys.executable, "main.py", "--knowledge-file", str(fixture_path),
    "--material-plan-mode", MATERIAL_PLAN_MODE,
    "--material-review-mode", MATERIAL_REVIEW_MODE,
    "--science-mode", SCIENCE_MODE,
    "--initial-image", str(INITIAL_IMAGE), "--experiment-name", experiment_slug,
    "--output-dir", str(OUT), "--generator-script", "kaggle_wan_wrapper.py",
    "--generator-executable", sys.executable, "--generator-working-dir", str(pipeline),
    "--wan-repo-dir", str(ROOT / "Wan2.2"), "--wan-model-preset", "i2v-a14b",
    "--continuity-mode", "chain", "--wan-max-area", str(SETTINGS["area"]),
    f"--generator-arg=--sampling-mode={SAMPLING_MODE}", f"--generator-arg=--base-steps={BASE_STEPS}", "--generator-arg=--gpu0-memory-gib=5",
    "--generator-arg=--gpu1-memory-gib=11", "--wan-memory-telemetry"]
command.extend([f"--generator-arg=--cpu-expert-cache-gib={CPU_EXPERT_CACHE_GIB}",
                f"--generator-arg=--prompt-cache-dir={PROMPT_CACHE_DIR}"])
if EXPERT_CACHE_DIR:
    command.append(f"--generator-arg=--expert-cache-dir={EXPERT_CACHE_DIR}")
if USE_VAE_CPU_STREAMING:
    command.append("--generator-arg=--vae-stream-to-cpu")
if STREAM_EXPERT_TO_GPU:
    command.append("--generator-arg=--stream-expert-to-gpu")
for area in SETTINGS["fallbacks"]:
    command.extend(["--wan-fallback-area", str(area)])
if EXPERT_SOURCE_DIR:
    command.append(f"--generator-arg=--expert-source-dir={EXPERT_SOURCE_DIR}")
if PROFILE_STAGES:
    command.append("--generator-arg=--profile-stages")
started = time.perf_counter()
subprocess.run(command, cwd=pipeline, check=True)
generation_wall_seconds = time.perf_counter() - started
print(f"Pipeline wall time including worker startup: {generation_wall_seconds:.1f}s")''')
md('''### Recover a preview import without regenerating
Every cell importing project modules restores the notebook kernel's source path.
If previewing an existing run reports `No module named frame_extractor`, add
`import sys; sys.path.insert(0, "/kaggle/working/image-to-video-pipeline")`
above its imports and rerun that preview cell. The successful generation subprocess
does not establish the notebook kernel's import path. Existing MP4s remain usable;
the preview, handoff extraction and composition use CPU processing.
''')
code('''from IPython.display import Video, display
from PIL import Image
import json
from frame_extractor import FrameExtractor
experiment_dir = OUT / experiment_slug
video_paths = [experiment_dir / f"{item['name']}.mp4" for item in clips]
for path in video_paths:
    assert path.is_file() and path.stat().st_size > 0, path
    display(Video(str(path), embed=True, width=480))
    print(json.loads(path.with_suffix(".generation.json").read_text()))
probe_source = video_paths[2] if len(video_paths) > 2 else video_paths[0]
for mode in ("near_end", "final", "mid"):
    image_path = OUT / f"review_handoff_{mode}.png"
    FrameExtractor().extract_handoff_frame(probe_source, image_path, mode=mode)
    print(mode, json.loads(Path(str(image_path) + ".json").read_text()))
    display(Image.open(image_path))''')
md('''## Compose at a readable speed
This step uses the real output paths, orders each procedure / clip / observation explicitly, and normalizes every segment to the same dimensions, frame rate and encoding before concatenation. It prevents mixed-resolution fallback clips or cards from breaking playback. Slower presentation duplicates frames; it does not invent intermediate chemical states.''')
code('''from imageio_ffmpeg import get_ffmpeg_exe
from stitcher import stitch_clips
segments = []
def card_segment(png):
    output = png.with_suffix(".mp4")
    subprocess.run([get_ffmpeg_exe(), "-y", "-loop", "1", "-i", str(png), "-t", "3",
        "-vf", "scale=1280:720,setsar=1,fps=16", "-an", "-c:v", "libx264", "-preset", "fast",
        "-crf", "18", "-pix_fmt", "yuv420p", str(output)], check=True, capture_output=True)
    return output
segments.extend(card_segment(png) for png in title_cards)
for index, raw_video in enumerate(video_paths):
    presented = raw_video.with_name(raw_video.stem + "_presented.mp4")
    filters = f"setpts=(PTS-STARTPTS)/{PLAYBACK_RATE},scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=16"
    subprocess.run([get_ffmpeg_exe(), "-y", "-i", str(raw_video), "-vf", filters, "-an",
        "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p", str(presented)], check=True, capture_output=True)
    segments.extend(card_segment(png) for png in procedure_cards[index])
    segments.append(presented)
    segments.extend(card_segment(png) for png in observation_cards[index])
segments.extend(card_segment(png) for png in conclusion_cards)
final = stitch_clips(segments, OUT / "science_educational_final.mp4")
print("Final:", final)
display(Video(str(final), embed=True, width=624))''')
md('''## Benchmark this run without generating again
This report counts actual generated seconds, not slower playback or cards. Scientific checks stay pending until reviewed in `scientific_review.json`. For controlled 2/3/4-second GPU comparisons, see `BENCHMARK.md`; longer profiles are experimental and the matrix disables area fallback so OOM limits remain visible.''')
code('''subprocess.run([sys.executable, "benchmark.py", "collect", "--knowledge-file", str(fixture_path),
    "--experiment-dir", str(experiment_dir), "--report-dir", str(OUT / "benchmark_report"),
    "--pipeline-wall-seconds", str(generation_wall_seconds)], cwd=pipeline, check=True)
print("Review:", OUT / "benchmark_report" / "scientific_review.json")''')
nb = {"nbformat": 4, "nbformat_minor": 5, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.12"}}, "cells": cells}
destination = OUTPUT / "Wan22_Kaggle_Continuous_State_Styled_Cards.ipynb"
destination.write_text(json.dumps(nb, indent=1), encoding="utf-8")
(REPO / destination.name).write_bytes(destination.read_bytes())
for index, cell in enumerate(cells):
    if cell["cell_type"] == "code":
        compile(cell["source"], f"notebook_cell_{index}", "exec")
print(f"Built {destination}; {len(cells)} cells; complete pinned snapshots; all Python cells compile")
