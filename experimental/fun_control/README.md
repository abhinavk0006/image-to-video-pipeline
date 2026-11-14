# Experimental Wan2.2-Fun A14B Control branch

Branch: feature/fun-control-motion-guides. The persistent-worker branch remains the
original pipeline. This branch is a separate experimental backend, not a confirmed
replacement on two T4 GPUs. No 5B model is selected.

## Review movements

Open ../../research/motion_review/reviewer.html locally in a browser. Play/pause,
scrub and step frames, change speed and camera angles, then download notes or tell
us the movement name and timestamp. There are 24 draft action families, including
aspiration (intentional immersion) separately from dispensing (tip above rim).
The apparatus shapes are generic wireframes and the timings are proposed, not
measured NCERT operations. Hands, grips, liquid dynamics, dosing and reactions
need separate review. This is not a claim to cover every possible laboratory motion.

Draft synthetic line guides are exported as four overlapping 33-frame, 16-FPS
chunks per eight-second template under research/motion_review/guides. These have
no text/axes/path arrows. They are synthetic outline candidates, not calibrated
depth maps, human skeletons or proven in-distribution Canny inputs. Review before
model generation. Each chunk represents one time interval, not an independently
restarted experiment. The backend's job list is independent; automatic reviewed
chunk chaining into a complete scientific procedure remains to be integrated.

## Runtime

Use Wan22_Fun_Control_A14B_Motion_Test.ipynb in a fresh Kaggle session, separate
from the original pipeline's dependency environment. It pins VideoX-Fun revision
991863c8c3fb9da946f21b675e52e45b0b53ed68 and our own branch snapshot. The runner
checks that revision and transforms its inference example through an audited AST
adapter rather than editing the upstream checkout. It accepts only hash-bound,
explicitly approved control/reference pairs. Motion approval is not control-input
or scientific approval.

Implemented experiments:
- FP16 compute for T4; sequential per-layer CPU offload; both A14B experts.
- Persistent models across an independent job batch and a bounded two-entry CPU
  cache of positive/negative text embeddings. The encoder is still constructed;
  this is not a lazy-T5 or cross-session prompt-cache implementation.
- Opt-in per-output-channel INT8 storage for Linear layers, keeping selected
  control/conditioning/input/output projections at full precision. Quantization
  uses a one-tensor-at-a-time safetensors conversion before offload hooks.
- Each layer dequantizes for FP16 F.linear. No fused INT8 speed claim. A content-addressed converted cache is populated one tensor at a time and
  loaded through mmap-backed CPU tensors. Both source checkpoint and converted
  shards are hashed; changed input, config or policy invalidates the cache.
  Source files and converted INT8 storage can need roughly 100 GB combined;
  check actual disk availability and mount prepared caches where necessary.
- Explicit quality reference settings: CFG=6 and 50 steps, no Lightning LoRA,
  TeaCache, CFG skipping, compilation or unreviewed FP4 by default.
- Strictly descending, same-aspect OOM-only area retries in fresh processes, with all attempts preserved; ordinary errors never trigger fallback.
- Unique output directory per invocation, per-job source/control hashes, seeds,
  wall time, device memory peaks, failures and preserved raw MP4s.

Not yet ported/validated: custom two-T4 expert streaming/placement, bounded
high-expert conversion cache, direct GPU
conversion, temporal VAE CPU streaming, full research/VBench
wrapper integration, and model-output scientific acceptance. The existing VBench adapter is now
called to prepare raw-video research inputs per successful job; evaluator scores
remain absent unless the official evaluator is installed and explicitly executed. Upstream group
 offload plus quantization is explicitly unsupported by the pinned helper; do not
combine them. Two GPUs are detected but the initial runner uses one GPU plus CPU
offload. This is intentionally explicit rather than silently pretending the
original worker's memory mapper supports different Control weights.

The released model bundle is roughly 69 GB. FP16 loading of both experts plus
T5 can exceed Kaggle host RAM even if individual layers fit VRAM. The INT8
streaming/cache path reduces conversion overlap, but OS paging, T5 and activations
still require actual target-hardware profiling. Converted caches also consume disk. Model
startup is therefore a target-hardware experiment; passing CPU smoke checks is
not evidence of GPU feasibility. No model weights are downloaded by default.

## Approval and scoring

A control review needs decision=approved_control_input, kind=canny/depth/pose,
reviewer, control_sha256 and reference_sha256. Only set approval after inspecting
both the guide and reference, their geometry/alignment and the selected interval.
Do not approve just to bypass the gate. run.json records generated_pending_review;
it is telemetry, not a complete scientific or VBench evaluation. Keep the current
pipeline as a matched reference baseline. Test FP16 vs INT8 with identical approved
inputs, seeds and settings before enabling quantization for experiments.

Smoke: python experimental/fun_control/test_control.py. An isolated CPU-only PyTorch environment exercises actual INT8 layers and
the streaming-cache roundtrip, tamper rejection, prompt-cache keys and bounds.
The source-adapter test also compiles the actual pinned upstream script. Model-specific numeric equivalence and actual target-GPU runs remain pending.

Use run_with_fallback.py for OOM-only retries (optional --fallback-size 512x288).
No fallback runs by default. Every area remains a draft pending visibility/science
review. Selective import namespaces load only the pinned Wan Control family,
avoiding unrelated speech/image model dependencies without editing upstream files.
The focused requirements retain the session GPU PyTorch installation; install a
compatible torch/torchvision pair separately if the environment requires it.

CPU integration: python experimental/fun_control/test_upstream.py imports the
actual pinned Control classes and executes a tiny randomly initialized Wan
transformer before/after cached INT8 conversion. This tests adapter integration;
it is not a pretrained A14B or CUDA quality result. Without Triton installed,
unused sparse kernels fail explicitly if invoked; dense PyTorch SDPA remains
the actual upstream attention implementation.
