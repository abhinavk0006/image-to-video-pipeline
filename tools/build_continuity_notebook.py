"""Package the reviewed source changes into a self-contained Kaggle notebook."""
import base64
import hashlib
import io
import json
from pathlib import Path
import zipfile
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
BASES = {
    "image-to-video-pipeline": "aec8c094420bac9d673be9a278dd8c9e5fe31993",
    "Wan2.2": "cb7b638da5633cc9e7c95f64ee0eac10b69007cf",
    "Edu-video-gen-dataset": "53f5bb6fb96208af6c7b1a56492bd9ed1a962186",
}
for dependency in ("Wan2.2", "Edu-video-gen-dataset"):
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=SOURCES[dependency], text=True).strip()
    if actual != BASES[dependency]:
        raise RuntimeError(f"{dependency} must be at pinned base {BASES[dependency]}, got {actual}")
FILES = {
    "image-to-video-pipeline": ["material_plan.py", "material_review.py", "main.py", "config.py", "MATERIAL_BUDGETS.md", "material_state.py", "science_cards.py", "models.py", "knowledge_loader.py", "CONTINUITY_AND_CARDS.md", "frame_extractor.py", "pipeline.py", "prompt_builder.py",
        "edu_video_rag_adapter.py", "video_generator.py", "stitcher.py", "kaggle_wan_wrapper.py",
        "test_frame_handoffs.py", "test_video_generator.py", "knowledge/ammonia_litmus_observation.json",
        "SCIENCE_QUALITY_NOTES.md", "PROJECT_HANDOFF.md", "BENCHMARK.md", "benchmark.py",
        "benchmark_matrix.json", "test_benchmark.py", "evaluator.py"],
    "Wan2.2": ["quantization/int8_linear.py", "wan/modules/attention.py", "quantization/kaggle_generate_video.py",
        "wan/image2video.py", "tests/test_science_worker_optimizations.py"],
    "Edu-video-gen-dataset": ["scripts/build_rag_prompt_bundle.py", "tests/test_rendering_plans.py"],
}
payloads = {f"{repo}/{name}": (SOURCES[repo] / name).read_bytes()
            for repo, names in FILES.items() for name in names}
manifest = {"base_commits": BASES, "files": {name: hashlib.sha256(data).hexdigest() for name, data in payloads.items()}}
archive = io.BytesIO()
with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
    z.writestr("manifest.json", json.dumps(manifest, indent=2))
    for name, data in payloads.items():
        z.writestr(name, data)
bundle = archive.getvalue()
(OUTPUT / "wan_continuity_cards_update.zip").write_bytes(bundle)
cells = []


def md(source):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": source.strip() + "\n"})


def code(source):
    # subprocess cwd only affects the child. Each notebook cell importing local
    # modules must establish the kernel's path itself, including on partial reruns.
    modules = sorted(set(re.findall(
        r"^(?:from|import) (knowledge_loader|models|material_state|material_plan|science_cards|frame_extractor|stitcher)\b",
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


md('''# Persistent Wan worker: continuous material state and science cards

Use Kaggle **Internet + two T4 GPUs**. This notebook pins the three active feature branches to the inspected commits and applies an embedded source update. No separate source upload is needed. It leaves the original notebook and its outputs intact.

The default experiment is the same four-shot chloride demonstration. Choose `litmus` below for a separate paper-colour regression shot. That requires a reference image already showing **one tube, colourless liquid, and damp red litmus in forceps above the opening**. Reagent preparation belongs in earlier shots; it is not generated during the paper observation.

The default `safe_2s` profile retains the original 33-frame temporal workload because longer shots previously ran out of memory on this setup. Its name does not guarantee that every resolution fits. `experimental_3s` uses 49 frames at a lower area and still needs hardware validation. `PLAYBACK_RATE` changes presentation speed on CPU; it adds no denoising work and cannot repair an incorrect reaction. The four-step Lightning model uses CFG=1, so the positive prompt carries the spatial and scientific constraints.''')
code('''import base64, gc, hashlib, io, json, os, subprocess, sys, time, zipfile
from pathlib import Path
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
ROOT = Path("/kaggle/working")
EXPERIMENT = "chloride"  # "chloride" or "litmus"
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
MATERIAL_OVERRIDES = {}   # clip name -> source-backed liquid_state fields; no guessed quantities
assert EXPERIMENT in {"chloride", "litmus"} and 0 < PLAYBACK_RATE <= 1
OUT = ROOT / f"wan_continuous_state_{EXPERIMENT}_{PROFILE}_{SAMPLING_MODE}"
OUT.mkdir(parents=True, exist_ok=True)
print(SETTINGS, "playback rate:", PLAYBACK_RATE)''')
code('''SOURCE_UPDATE_B64 = "'''+base64.b64encode(bundle).decode()+'''"
update = zipfile.ZipFile(io.BytesIO(base64.b64decode(SOURCE_UPDATE_B64)))
manifest = json.loads(update.read("manifest.json"))
refs = {"Wan2.2": "feature/persistent-wan-worker", "image-to-video-pipeline": "feature/persistent-wan-worker", "Edu-video-gen-dataset": "feature/structured-reference-plans"}
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
for name, expected_hash in manifest["files"].items():
    relative = Path(name)
    assert relative.parts[0] in refs and not relative.is_absolute() and ".." not in relative.parts
    data = update.read(name)
    assert hashlib.sha256(data).hexdigest() == expected_hash
    destination = ROOT / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.read_bytes() != data:
        backup = destination.with_name(destination.name + ".before_science_update")
        if not backup.exists():
            backup.write_bytes(destination.read_bytes())
    destination.write_bytes(data)
pipeline = ROOT / "image-to-video-pipeline"
sys.path.insert(0, str(pipeline))
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(ROOT / "Edu-video-gen-dataset" / "requirements.txt"),
    "diffusers>=0.31,<0.33", "transformers>=4.49,<=4.51.3", "accelerate>=1.1.1", "easydict", "safetensors", "ftfy", "huggingface_hub", "imageio", "imageio-ffmpeg", "Pillow>=10.1", "matplotlib"], check=True)
(OUT / "source_manifest.json").write_text(json.dumps(manifest, indent=2))
print("Pinned bases plus verified embedded source update:", manifest["base_commits"])''')
md('''## Source quantities and material preflight
The current retrieved salt-analysis record does not specify reagent amounts or
vessel capacity. The fixture also has no authoritative quantities. Supply them
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
md('''## Build atomic shot plans
The chloride plan uses one starting image, then the **actual final decoded frame of the preceding generated shot** for every continuing operation. A named material-state track prevents static reference images from resetting the tube. These frames carry the generated state; they still need scientific review. Declare `starts_new_sample` with a separate image when a procedure intentionally changes aliquot. No shot asks for pouring, swirling and indicator testing together.

For the optional litmus shot, the gas-producing preparation must already be complete. The fixture only evaluates the visible paper response. It remains a visual test and needs source and video review before instructional use.''')
code('''def unique_input(name):
    matches = list(Path("/kaggle/input").rglob(name))
    if len(matches) != 1:
        raise RuntimeError(f"Mount exactly one {name}; found {len(matches)}")
    return matches[0]

if EXPERIMENT == "litmus":
    fixture = json.loads((pipeline / "knowledge" / "ammonia_litmus_observation.json").read_text())
    INITIAL_IMAGE = unique_input("ammonia_litmus_ready.png")
    fixture["clips"][0]["input_frame_path"] = str(INITIAL_IMAGE)
    fixture["clips"][0]["duration_seconds"] = SETTINGS["seconds"]
else:
    DRY_IMAGE = unique_input("salt_analysis_one_test_tube_dry_salt.jpeg")
    INITIAL_IMAGE = DRY_IMAGE
    rag = ROOT / "Edu-video-gen-dataset"
    if not (rag / "rag_index_matrix.npz").exists():
        subprocess.run([sys.executable, "scripts/build_rag_index.py"], cwd=rag, check=True)
    bundle_path = OUT / "rag_bundle.json"
    subprocess.run([sys.executable, "scripts/build_rag_prompt_bundle.py", "Detect one cation and one anion in a salt", "--top-k", "1", "--output", str(bundle_path)], cwd=rag, check=True)
    rag_record = json.loads(bundle_path.read_text())["retrieved_experiments"][0]
    assert all(word in rag_record["title"].lower() for word in ("cation", "anion")), rag_record["title"]
    print("Retrieved:", rag_record["title"], rag_record.get("retrieval_score"))
    def shot(name, image, before, action, after, procedure, observation, chained=False):
        result = {
            "name": name, "input_frame_path": str(image) if image else None,
            "state_track": "chloride_sample_1", "tool_contact_policy": "above_rim",
            "liquid_state": {"level_change": "increase", "after_appearance": after},
            "scene": {"camera": "Locked close view; keep the existing test tube fixed"},
            "state_before": before, "action": action, "state_after": after,
            "duration_seconds": SETTINGS["seconds"], "continuity_required": chained,
            "reference_mode": "chained" if chained else "independent", "needs_reference_image": not chained,
            "reference_policy": "previous_state_keyframe" if chained else "state_keyframe",
            "target_object": "the contents of the single existing test tube",
            "preserve_objects": ["the existing test tube and holder", "the laboratory background"],
            "quantity_constraint": "Use this step's documented dose, not an equal dose for every reagent. Preserve the starting liquid and headspace.",
            "observable_target": after,
            "action_pacing": "Dispense only the specified dose, stop delivery, then show the documented endpoint throughout the contents. Use mixing only when the source calls for it; no time lapse.",
            "procedure_text": procedure, "observation_text": observation,
            "prompt_bundle": {"stop_condition": after, "negative_prompt": "extra tube, duplicate vessel, fast motion, camera movement"},
        }
        if chained:
            result.update(handoff_policy="final", handoff_entity="the same sample and meniscus in the existing tube")
        return result
    fixture = {"name": "One-Tube Chloride Salt Analysis", "subject": "Qualitative salt analysis",
        "educational_goal": "Observe chloride confirmation with silver nitrate and ammonia in a controlled one-tube visual sequence.",
        "metadata": {"manual_review_required": True, "manual_review_reasons": ["Controlled visual fixture: confirm procedure details against the source manual before instructional use."]},
        "rag_experiment": rag_record,
        "conclusion_text": "The white silver chloride precipitate dissolves in ammonia, supporting the chloride identification.",
        "clips": [
            shot("prepare_unknown_salt", DRY_IMAGE, "One test tube contains white salt.",
                "Slowly add distilled water from one dropper into the existing test tube; keep the tube stationary.",
                "The salt dissolves in the water in that same tube, leaving a clear colourless solution.",
                "Add distilled water to the salt.", "The salt dissolves to form a clear solution."),
            shot("acidify_unknown_solution", None, "The existing test tube contains clear colourless salt solution.",
                "Slowly add dilute nitric acid from one dropper into the existing tube.",
                "The solution remains clear and colourless.", "Acidify the solution with dilute nitric acid.", "The solution remains clear and colourless.", chained=True),
            shot("silver_nitrate_chloride_test", None, "The existing tube contains clear colourless acidified salt solution.",
                "Slowly add dilute silver nitrate from one dropper into the existing tube.",
                "A white curdy precipitate forms and stays visible inside that same tube.",
                "Add dilute silver nitrate to the acidified solution.", "A white curdy silver chloride precipitate forms.", chained=True),
            shot("ammonia_dissolves_chloride_ppt", None, "The same tube contains the white precipitate visible in the preceding shot.",
                "Slowly add dilute ammonia from one dropper into the same tube.",
                "The existing white precipitate dissolves, leaving a clear colourless solution in the same tube.",
                "Add dilute ammonia to the silver chloride precipitate.", "The precipitate dissolves; the solution becomes clear.", chained=True),
        ]}
clips = fixture["clips"]
if EXPERIMENT == "chloride":
    clips[0]["liquid_state"].update(solid_change="The original salt grains visibly shrink and disappear", after_clarity="clear", after_colour="colourless", after_solid="none", reaction_scope="whole_contents")
    clips[0]["liquid_state"]["after_appearance"] = "optically transparent colourless liquid; the background is visible through the full column"
    clips[1]["liquid_state"]["after_appearance"] = "transparent colourless solution, no precipitate"
    clips[2]["liquid_state"]["after_appearance"] = "white curdy precipitate in the liquid, as documented"
    clips[1]["liquid_state"].update(after_clarity="clear", after_colour="colourless")
    clips[3]["liquid_state"].update(after_appearance="transparent colourless solution throughout", solid_change="The existing precipitate dissolves and disappears", after_clarity="clear", after_colour="colourless", after_solid="none", reaction_scope="whole_contents")
for item in clips:
    item["liquid_state"] = {**item.get("liquid_state", {}), **MATERIAL_OVERRIDES.get(item["name"], {})}
# Resolve tracks in the saved plan as well, so benchmark dependencies are explicit.
from knowledge_loader import _parse_clip
from models import Experiment
from material_state import prepare_state_tracks
from material_plan import compile_material_plan
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
experiment_slug = "one_tube_chloride_salt_analysis" if EXPERIMENT == "chloride" else "ammonia_litmus_observation"
fixture_path = OUT / "experiment.json"
fixture_path.write_text(json.dumps(fixture, indent=2))
print("Generated frames per shot:", max(17, 4 * __import__("math").ceil((SETTINGS["seconds"] * 16 - 1) / 4) + 1))
print("Shots:", [item["name"] for item in clips])''')
code('''from PIL import Image
from science_cards import render_cards
cards_dir = OUT / "cards"
def cards(name, heading, body, kind):
    return render_cards(cards_dir, name, heading, body, subject="chemistry", kind=kind,
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
code('''command = [sys.executable, "main.py", "--knowledge-file", str(fixture_path),
    "--material-plan-mode", MATERIAL_PLAN_MODE,
    "--material-review-mode", MATERIAL_REVIEW_MODE,
    "--initial-image", str(INITIAL_IMAGE), "--experiment-name", experiment_slug,
    "--output-dir", str(OUT), "--generator-script", "kaggle_wan_wrapper.py",
    "--generator-executable", sys.executable, "--generator-working-dir", str(pipeline),
    "--wan-repo-dir", str(ROOT / "Wan2.2"), "--wan-model-preset", "i2v-a14b",
    "--continuity-mode", "chain", "--wan-max-area", str(SETTINGS["area"]),
    f"--generator-arg=--sampling-mode={SAMPLING_MODE}", f"--generator-arg=--base-steps={BASE_STEPS}", "--generator-arg=--gpu0-memory-gib=5",
    "--generator-arg=--gpu1-memory-gib=11", "--wan-memory-telemetry"]
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
print(f"Built {destination}; {len(cells)} cells; embedded source {len(bundle):,} bytes; all Python cells compile")
