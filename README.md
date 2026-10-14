# Scientific Image-to-Video Generation Pipeline

An orchestrator pipeline that generates multi-clip videos from static initial images and structured scientific experiment JSON logs. It ensures visual continuity across transitions by feeding the final frame of the preceding clip as the initial frame of the next.

Designed to run in resource-constrained cloud environments (e.g., 62GB RAM / 46GB L40S GPU) using memory-efficient execution strategies.

The repository also includes an experimental Kaggle adapter for the two-T4,
streamed INT8 Wan 2.2 I2V-A14B Lightning runner in the sibling `Wan2.2` repo.
The adapter connects the pipeline's existing JSON-lines generator interface to
that runner; it does not make the 14B model suitable for a 2GB laptop GPU.

---

## 🚀 Key Features

* **Controlled Multi-Clip Generation**: Defaults to independent, verified reference images per clip; opt into final-frame chaining only when a physical action genuinely continues across shots.
* **Resume Support**: Interrupted runs automatically skip already generated clips, allowing seamless pipeline recovery.
* **Generator Daemon Interface**: Reuses one JSON-RPC Wan process across sequential clips. The two diffusion experts still swap to fit available VRAM.
* **Robust Frame Handoffs**: Sequentially decodes video frames without `ffprobe` or end-of-file seeks, supports mid-clip handoffs, and extracts only when a later clip needs the image.
* **Scientific Prompt Builder**: Carries structured target, quantity, preservation, pacing, and observation fields into generation prompts.

### Visual fidelity for science demonstrations

The prompt builder applies the same scientific visual rules to every experiment:
perform one procedure step at a time; act on the named target while preserving
other apparatus; show an observation on the physical object where it occurs;
keep any named indicator, strip, or measuring device visible; follow the
manual's sequence and quantities; and avoid inventing reactions or apparatus.
For paper or strip indicators, the change belongs on the paper. A liquid
indicator changes the sample only when the procedure specifies that result.
When the source omits an amount, prompts avoid inventing an exact measurement
or ratio and keep vessels visibly below the rim. These rules improve guidance,
but generated clips still require visual and scientific review.

---

## 🛠️ Installation & Setup

Set up the workspace and download dependencies using the provided configuration script:

```bash
# Run minimal installation script
chmod +x lightning_setup_minimal.sh
./lightning_setup_minimal.sh
```

---

## 🏃 Execution

### Kaggle: Wan 2.2 I2V-A14B (experimental)

Use a Kaggle notebook with **Internet enabled and both Tesla T4 GPUs enabled**.
Clone all three repositories into `/kaggle/working`, mount your starting image,
then run from the pipeline checkout:

```bash
cd /kaggle/working
git clone https://github.com/abhinavk0006/Wan2.2.git
git clone https://github.com/abhinavk0006/image-to-video-pipeline.git
git clone https://github.com/abhinavk0006/Edu-video-gen-dataset.git
python -m pip install -q -r /kaggle/working/Edu-video-gen-dataset/requirements.txt imageio-ffmpeg
cd /kaggle/working/image-to-video-pipeline
python main.py \
  --query "YOUR EXPERIMENT QUERY" \
  --rag-repo-dir /kaggle/working/Edu-video-gen-dataset \
  --rag-top-k 1 \
  --rag-max-clips 1 \
  --rag-clip-duration-seconds 1 \
  --initial-image /kaggle/input/YOUR_DATASET/YOUR_IMAGE.jpeg \
  --output-dir /kaggle/working/pipeline_outputs \
  --generator-script kaggle_wan_wrapper.py \
  --generator-working-dir /kaggle/working/image-to-video-pipeline \
  --wan-repo-dir /kaggle/working/Wan2.2 \
  --wan-model-preset i2v-a14b
```

The `--query` path calls the dataset repo's TF-IDF retriever and source-grounded
prompt-bundle builder; its best positive match becomes a pipeline experiment,
with source IDs and source-review status retained in the generated knowledge
JSON. Draft or generic source details are surfaced as `SOURCE REVIEW REQUIRED`
before generation. Add `--require-reviewed-source` to stop before GPU work until
the experiment and its procedure/observations have reviewed status. Per-step
local `starting_image` paths are used as explicit keyframes when mounted; remote
image URLs are not fetched. `--rag-max-clips 1` limits this initial integration preview to one
retrieved procedure step; omit the flag to generate all retrieved steps. The
supplied image sets the first clip; later clips use each preceding
clip's final frame. Dataset image URLs are not downloaded or substituted for
your mounted starting image. To run the neutral one-clip test-tube preview
without retrieval, replace `--query` and `--rag-repo-dir` with
`--knowledge-file knowledge/test_tube_drop.json`.

The adapter maps each clip duration to Wan's 16-fps `4n+1` frame count
(minimum 17 frames), and uses the four-step Lightning adapters with the
streamed INT8 experts. `--rag-clip-duration-seconds 1` is the currently tested
preview duration; longer durations are available but still need Kaggle VRAM
and runtime validation.

For chained shots, handoff timing is configurable per clip. The default
`handoff_mode` is `near_end`, which extracts `0.5` seconds before the end to
avoid relying on an end-of-file seek. Use `handoff_mode: "mid"` when
the meaningful state occurs halfway through a clip. Use `handoff_mode: "final"` when
the final state is the relevant handoff, or use
`handoff_mode: "offset"` with `handoff_offset_seconds` when the meaningful
state occurs at a known earlier time:

```json
{
  "name": "release_one_drop",
  "handoff_mode": "offset",
  "handoff_offset_seconds": 0.75
}
```

Choose the handoff frame by inspecting the generated clip: it must visibly
contain the object and state named by the next clip. Use an earlier offset when
the object disappears or the final frames introduce a new action; use
`final` only when the final state is stable. The timing controls do not make
Wan recover an object that is absent from the selected frame.

The pipeline defaults to independent shots. With `--continuity-mode independent`,
each clip uses its explicit `input_frame_path` when present, otherwise the
verified `--initial-image` reference. This avoids compounding generated-frame
drift across a long experiment. Use `--continuity-mode chain` only for actions
that must continue visually, such as a single pour or stirring motion. Per-step
durations can be set with `clip_duration_seconds` in manual JSON; RAG records
may provide `duration_seconds`.

For visual review, the persistent-worker Kaggle notebook defaults to a two-clip
indicator pilot, previews mounted per-step reference images before GPU work, and
shows generated clips and extracted handoff frames afterward. Review the actual
target and observation before setting `RUN_LITMUS_PILOT_ONLY = False` for the
full sequence. A text prompt cannot ensure that a video model obeys scientific
visual constraints; use a reviewed keyframe when a critical object or state
change must be exact.

For activation-memory experiments, pass `--wan-max-area 345600` (or another
positive pixel area) to the Kaggle command and `--wan-memory-telemetry` to log
allocated, reserved, and free memory on each visible GPU. If a requested area
causes CUDA OOM, configure one or more lower areas in descending order, for
example `--wan-max-area 399360 --wan-fallback-area 200704
--wan-fallback-area 129024`. The same clip is retried through the existing
daemon without restarting the pipeline; reference and chained-frame inputs are
unchanged. Only CUDA OOM failures trigger this fallback, and logs identify the
selected area (or the final area on failure).

The daemon fallback contract is intentionally strict: the Wan child must write
its CUDA OOM traceback/diagnostic to stderr and exit nonzero if it dies before
returning a JSON response. The Kaggle wrapper forwards that stderr and reports
the child exit code; the pipeline retries only when the combined diagnostics
contain a recognized CUDA OOM marker. A bare signal, nonzero exit code, or
ordinary crash is fatal and is not retried. No Wan2.2 code change is required
unless its daemon suppresses CUDA OOM diagnostics; in that case it must preserve
the exception text on stderr and return a nonzero exit code (or return the
existing `{"status":"error","error":"..."}` JSON response).

**Runtime and validation:** use the persistent Wan daemon for multi-clip Kaggle
runs. It keeps the VAE and process alive, releases T5 after prompt encoding to
reduce host-memory pressure during expert conversion, and restores T5 when the
next clip needs it. Switching noise phases still evicts and reloads the other
expert to stay within two T4 GPUs; repeated expert conversion remains a major
runtime cost and is measured in the Wan logs. The daemon keeps stdout reserved
for `READY` and one JSON response per request, with diagnostics on stderr.
Resume support skips clips whose output files already exist.

Outputs are written under `/kaggle/working/pipeline_outputs/<experiment>/`;
download `final_video.mp4` from Kaggle's Output panel after the run.

Run the pipeline by providing the path to your scientific experiment JSON log, the generator wrapper script, the Wan 2.2 model directory, and the initial seed image:

```bash
python main.py \
  --knowledge-file knowledge/acid_base_titration.json \
  --generator-script wan_local_wrapper.py \
  --wan-repo-dir /teamspace/studios/this_studio/Wan2.2 \
  --wan-model-preset i2v-a14b \
  --initial-image outputs/acid_base_titration/input/input_image.jpg
```

### Parameters:
* `--knowledge-file`: Absolute path to the scientific experiment JSON configuration.
* `--generator-script`: Script wrapping the underlying generator (defaults to `wan_local_wrapper.py`).
* `--wan-repo-dir`: Path to the cloned `Wan2.2` repository containing your model code.
* `--wan-model-preset`: The model preset to run (`i2v-a14b` or `ti2v-5b`).
* `--initial-image`: Starting image for the very first clip.

---

## 🧠 Memory Optimizations

To run the **14B Image-to-Video** model without crashing under standard hardware limits:
1. **Lazy Weights Loading**: High-noise and low-noise models are loaded sequentially on demand rather than all at startup.
2. **CPU-to-GPU Memory Unloading**: The inactive model is explicitly offloaded back to the CPU and garbage collected before the active model is loaded, keeping memory overhead within physical RAM boundaries.
3. **Low-Precision Execution**: Models are converted to `bfloat16` and run with `offload_model=True` to minimize VRAM footprint.

The original `wan_local_wrapper.py` targets its separate local Wan setup. For Kaggle's two-T4 streamed INT8 path, use `kaggle_wan_wrapper.py` and the instructions above; it does not use the local wrapper's VRAM assumptions.

## Structured state/action inputs

Natural-language `--query` requests use the local `knowledge/` RAG records first.
Only a miss loads the Qwen fallback. The fallback is normalized into the same
experiment/video contract as stored records: clip duration, state/action,
reference, handoff, `procedure_text`, and `observation_text` are retained for
later deterministic card rendering. Invalid JSON or a missing clip list fails
loudly instead of silently producing a clips-only plan.

New experiment inputs can describe each step explicitly:

```json
{
  "state_before": "colorless solution in flask",
  "action": "add reagent slowly",
  "state_after": "pale pink endpoint",
  "visual_priority": "high",
  "continuity_required": false,
  "reference_policy": "state_keyframe",
  "duration_seconds": 2
}
```

These fields are compiled into the motion prompt and override global chaining
when `continuity_required` is present. `state_keyframe` and `independent`
policies use the verified reference image; continuous actions can set
`continuity_required` to `true` and use a handoff frame.

For reliable multi-step experiments, continuous steps should also name their
source explicitly with `reference_clip`, for example:

```json
{
  "name": "ammonia_dissolves_chloride_ppt",
  "state_before": "white curdy silver chloride precipitate already visible",
  "action": "add dilute ammonia gradually",
  "state_after": "clear colorless solution",
  "continuity_required": true,
  "reference_policy": "previous_state_keyframe",
  "reference_clip": "silver_nitrate_chloride_test"
}
```
