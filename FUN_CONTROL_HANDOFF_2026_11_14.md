# Wan2.2-Fun A14B Control handoff

Branch: `feature/fun-control-motion-guides`.
Notebook: `Wan22_Fun_Control_A14B_Motion_Test.ipynb`.
The original continuous-state notebook remains on `feature/persistent-wan-worker`.

## Review first

Open `research/motion_review/reviewer.html` locally. It is a standalone offline
viewer with 24 motion templates, camera controls, slow playback, frame stepping
and downloadable notes. Review decisions are bound to the motion-data hash.
There are 24 MP4 previews and 96 overlapping 33-frame synthetic-outline chunks.
The eight-second paths are illustrative proposals, not measured procedures.
Each chunk spans two seconds of path time; its 33 frames at 16 FPS play for 2.0625
seconds. Adjacent chunks share their boundary frame.

Candidate mappings cover 210 chemistry source steps heuristically. All need
apparatus fitting and review; one membrane-construction step explicitly requires
a new template. Matching a keyword does not establish coverage of a complete
procedure. Hands, grips, fluid simulation, calibrated dosing, deformable objects,
reaction chemistry and validated circuit topology are not implemented by these
wireframes. Outline guides are not calibrated depth/pose controls.

## What this backend does

- Uses the pinned VideoX-Fun A14B Control implementation and both noise experts.
- Keeps models loaded across a batch; bounds CPU prompt-embedding reuse.
- Supports FP16 or experimental per-channel INT8 Linear storage with protected
  control/conditioning projections, sequential CPU offload and layer-local
  dequantization. A content-addressed, hashed, mmap-backed conversion cache avoids
  materializing both full FP16 expert state dictionaries at once.
- Preserves every failed attempt and retries only actual CUDA OOM errors at
  explicitly configured smaller areas of the same aspect ratio.
- Requires a separate hash-bound reference/control review before generation.
- Records seeds, prompts, hashes, raw outputs, elapsed time and CUDA memory peaks;
  prepares inputs for the existing research evaluator without inventing scores.

The initial profile uses one GPU plus CPU offload. The original custom two-T4
expert placement and temporal VAE CPU streaming are not ported. No Lightning
adapter, fused INT8 speedup or automatic scientific acceptance is claimed. It
does not yet chain the motion chunks through reviewed image handoffs automatically.

## Verification completed on this laptop

- Seven CPU smoke tests passed, including actual Torch INT8 layers, bounded
  positive/negative prompt-cache keys, review rejection, converted-cache reuse
  and tamper rejection, fallback geometry and the actual source AST contract.
- Actual pinned Control classes import successfully using focused package imports.
- A tiny randomly initialized actual Wan transformer executes before/after cached
  INT8 conversion on CPU. Nonzero-output maximum absolute difference: about
  0.00057. This is an integration check, not an A14B accuracy benchmark.
- The offline viewer passed headless Chrome playback, scrubbing, camera, review
  JSON download and 375-pixel layout checks with all 24 options present.
- All 24 preview MP4s and 96 guide MP4s decode fully. Notebook cells compile.

Isolated CPU integration versions: Python 3.12, Torch 2.15.0+cpu,
torchvision 0.29.2+cpu, diffusers 0.43.0, transformers 5.23.0,
accelerate 1.15.0, safetensors 0.8.0. These versions are not a validated CUDA
environment; retain a compatible GPU Torch/torchvision pair in Kaggle.

## Target-hardware validation still required

The laptop's MX450 with 2 GB VRAM cannot validate A14B generation. No pretrained
model weights or generated scientific outputs were tested here. The model bundle
is roughly 69 GB; source plus converted caches can require roughly 100 GB of
storage. Host paging, T5, attention and VAE activations still need profiling.
Neither two-T4 fit nor a generation-time improvement is established.

Use a fresh Kaggle session. First review a motion and its alignment to the exact
reference image, then enable dependency installation and provide mounted model
files or explicitly enable downloading. The notebook defaults to review-only;
set the reviewer, reference, control approval and RUN_MODEL after review. Compare
FP16 and INT8 at identical inputs/settings when hardware permits. Inspect the
resulting science before using any clip in the paper. Research scores remain
unset until the official evaluators and scientific reviews are actually run.
