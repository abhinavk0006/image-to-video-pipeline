# Reference-image plan, runtime audit and paper benchmarking

## Reference images for every experiment family

Start with a selected source protocol, not the broad experiment title. Each shot
needs: apparatus IDs, exact connections/geometry, sample identity, starting state,
source-defined appearance, quantity facts/unknowns, action, observation, framing
and source citation. Review these before asking an image model to draw anything.
An image cannot independently establish an exact volume, reagent identity or
concentration. Keep source measurements separate from visual appearance.

| Family | Reference method | Required review |
|---|---|---|
| Wet chemistry | One reviewed apparatus master photo/render; edit its prepared sample state | Vessel count, clarity/colour, meniscus, reagent/sample IDs, headspace |
| Titration and measurement | Geometrically constrained setup; deterministic scale/readout overlays | Burette/pipette arrangement, initial readings, scale calibration |
| Circuits/electronics | Validated node/component graph to SVG/PNG; optional matching apparatus render | Connections, junctions, polarity, meter placement, textbook layout |
| Optics | Geometry-derived apparatus and ray diagram; photo only where appropriate | Axis, lens/mirror placement, rays, measured positions |
| Mechanics | Explicit dimensions/constraints; deterministic diagram/animation for geometric relationships | Pivot/contact points, forces, trajectories and coordinate meaning |
| Thermal/fluid experiments | Reviewed apparatus scene plus source-defined initial conditions | Connections, interfaces, instruments, phase/colour and timing |
| Abstract processes | Clearly labelled deterministic schematic/animation | Distinguish explanatory models from visible observations |

Workflow:
1. Audit catalogue by family and source completeness. Unresolved physics records
   wait for the manual. Multi-variant records require a selected protocol.
2. Build a small apparatus library with persistent IDs, camera geometry, neutral
   backdrop and reference-image provenance. Do not reuse a photograph as proof of
   unmeasured quantities. Keep each genuinely different apparatus/setup distinct.
3. Generate/edit a few candidate master references for each apparatus family.
   Prefer edits of an approved master for the same apparatus to reduce visual drift.
4. Draw quantitative scales, labels, circuit wires and rays deterministically.
   Match an NCERT layout after checking the specific source figure and permitted
   reuse. Photorealism alone does not establish connection correctness.
5. Review apparatus count, topology, starting state, useful contrast and whether
   the action and observation are visible at the final generation resolution.
6. Save immutable versions: experiment/protocol/shot IDs, asset SHA256, prompt,
   generator/version, source page, expected state, reviewer and approval status.
   Crop/region annotations belong to this exact image hash.
7. Generate new references only for the initial setup, a fresh sample, a camera
   change or an intentional apparatus change. Same-sample continuations consume
   the previous generated state, ideally selected through reviewed evidence.
8. Pilot representative chemistry, measurement and physics families before bulk
   generation. Batch only approved specifications; no need to generate a separate
   unrelated master image for every procedure step.

## Runtime audit: more optimization remains

Existing: persistent daemon; four-step Lightning with CFG=1; FP16 DiT; weight-only
INT8 storage; prompt pre-encoding; bounded high-expert CPU reuse; optional converted
expert/prompt disk caches; optional VAE output streaming and direct GPU conversion;
resolution fallback; selective resume and repair. None is a measured optimal policy.

| Finding | Next work | Trade-off / validation |
|---|---|---|
| INT8 expands each layer to float before F.linear | Compare a compatible fused weight-only kernel | Integration, device hooks, numeric error and scientific output must be checked on T4; do not assume speedup |
| Startup loads/hashes T5 even with warm prompt caches | Lazy T5 construction after validating model/tokenizer cache identities | Preserve content-sensitive invalidation; cold/mixed cache paths need regression and GPU checks |
| Restoring T5 replaces encoder wrapper while local reference remains stale | Fixed by refreshing local encoder after restoration | CPU control-flow regression; full model restoration still needs GPU/session test |
| Expert loads dominated the old reported run | Measure CPU cache retention/hits, mounted sources and optional converted disk cache first | Host RAM/disk can prevent hits; two experts need about 27 GiB on disk |
| Output CPU streaming addresses decode memory | Compare streaming on/off and consider spatial VAE tiling separately | Does not solve attention activation OOM; tiling must preserve temporal caches and avoid seams |
| Attention uses SDPA fallback when FlashAttention unavailable | Log actual backend eligibility and profile dispatch | Fused backend support depends on GPU, dtype, shapes and installed version |
| Two-GPU module placement may add transfers | Compare balanced/contiguous, then consider topology-aware placement | Retain activation headroom; do not simply increase weight budgets |
| Offload/empty_cache/synchronization calls may cost time | Profile call frequency before selectively removing redundant calls | Some calls are necessary at expert/VAE boundaries to fit memory |
| Recompilation/model swaps can erase compile gains | Try regional compilation on stable blocks only after other bottlenecks | Include compilation cost and memory; do not blindly compile the entire pipeline |
| Handoffs and benchmark collection decode clips repeatedly | Consider a single-pass multi-frame extraction API | Primarily CPU savings; compare exact selected frame indices |
| Slow or invisible scientific actions cause rerenders | Improve source specification and visibility; isolate action and observation | Measure accepted scientific seconds per wall time, not only raw GPU FPS |
| Approx. 2-second clips rush multiple actions | Atomic actions and duration sweeps at fixed source plans | Slow playback adds no native motion; interpolation/upscaling cannot repair chemistry |

Priority: confirm baseline/cache behaviour -> lazy encoder/source reuse -> compatible
kernel and placement comparisons -> VAE/attention memory -> controlled longer clips.
Do not batch dependent clips by expert phase: each continuation needs the previous
clip's decoded/reviewed state. Independent experiments could be scheduled differently,
but storing multiple latents/activations and delaying validation changes the budget.

## Automatic capture now implemented

CLI `main.py` and the continuous-state notebook capture a unique report for each
attempt, including interrupted/failed runs, through `generation_benchmark.py`.
Notebook launches mark the child as already managed to prevent duplicate capture.
Direct calls to ExperimentPipeline or standalone Wan generation scripts are lower
level APIs and are not automatically enrolled by this wrapper.

Reports contain command, source-plan snapshot/hash, installed package versions,
existing-before-run clips, generation status, measured invocation wall time,
technical/scientific review report, missing clips and raw video hashes. Existing
clips make the invocation ineligible for cold-generation timing. Failures stay in
the denominator. Resumed timing must be reported separately from fresh generation.

Official VBench custom-input preparation uses only raw generated clips. If the
`vbench` executable is installed, scoring is executed after generation; otherwise
status is `prepared_not_scored`, with exact commands saved. No fake score fills a
missing model/dependency or failed metric. Evaluation wall time is separate from
generation time. The six supported dimensions are subject consistency, background
consistency, motion smoothness, dynamic degree, aesthetic quality and imaging
quality. These are raw metrics, not percentage realism or chemical accuracy. High
motion is not automatically desirable for slow laboratory work. A static video
can score consistently while missing the required procedure.

Install/pin the official VBench environment and its dependencies/checkpoints in a
separate evaluation setup following its documentation. To score a prepared bundle:
run the commands saved in `vbench/metric_run.json`, from separate per-dimension
output folders. Preserve official result files, tool/checkpoint versions and logs.
Do not install competing Torch versions into the working generation environment.
I2V reference fidelity/camera metrics are a distinct extension requiring aligned
reference-image metadata; they are not silently claimed by the six-metric adapter.

## Paper protocol

Three score groups, reported separately:
- Visual generation quality: the six official custom-input VBench dimensions;
  optional separately validated VBench-I2V reference fidelity and camera metrics.
- Scientific/educational quality: source-backed step correctness, apparatus count
  and continuity, dose/level behaviour, observation target and extent, causal order,
  visibility, pacing and whole-sequence success. Use timestamped evidence from two
  blinded independent reviewers and adjudicate disagreements. Report denominators,
  pending reviews, inter-rater agreement and uncertainty; learning gains need a
  separate learner study and cannot be inferred from visual scores.
- Efficiency/reliability: fresh and resumed wall time, cold/warm cache condition,
  native generated seconds, accepted scientific seconds per wall second, GPU/host
  memory, OOM/failure rates and repair/evaluation cost. No double counting expert
  load time already nested in denoising.

Pilot a stratified set of about 12 source-complete experiments with at least three
fixed seeds per condition, then plan the final sample size from observed variation
and the paper's claim. Include plain Wan with matched inputs, the earlier pipeline,
the full system and ablations removing individual controls. Where available, add
an appropriate external generator under disclosed budgets. Use held-out families,
shared sources/seeds/native resolution and clip lengths for comparisons. Report
means/distributions and confidence intervals clustered by experiment rather than
treating correlated frames as independent samples. This pilot count is a planning
suggestion, not a claim of statistical adequacy.

Evaluate raw clips separately from the final cards/composed lesson. Evaluate long
sequence transitions and complete procedure success separately from per-clip mean
quality. Do not report the six custom-input dimensions as a VBench leaderboard total.
FVD needs meaningful reference and generated video distributions; do not calculate
or interpret it from a handful of clips. PSNR/SSIM are not primary metrics for
unpaired generative science videos. A VLM score must be calibrated against expert
review, including deliberately incorrect challenge footage, before being used as
an automatic scientific verifier. No benchmark results have been invented here.

## Sources checked

- Official VBench: https://github.com/Vchitect/VBench
- Official I2V protocol: https://github.com/Vchitect/VBench/blob/master/vbench2_beta_i2v/README.md
- PhyEduVideo: https://arxiv.org/abs/2601.00943 (physics explanatory concepts; not automatically a sequential chemistry benchmark)
- EduVQA: https://arxiv.org/abs/2603.03066 (early-mathematics concept assessment; requires domain validation for chemistry)
- PyTorch SDPA: https://docs.pytorch.org/docs/main/generated/torch.nn.functional.scaled_dot_product_attention.html
- Diffusers memory: https://huggingface.co/docs/diffusers/optimization/memory (techniques require adaptation to this custom Wan runtime)
- PyTorch compile: https://docs.pytorch.org/docs/stable/generated/torch.compile
