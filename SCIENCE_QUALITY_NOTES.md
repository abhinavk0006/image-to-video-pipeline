# Persistent worker quality update — 18 October 2026

Use `Wan22_Kaggle_Science_Quality_Persistent_Worker.ipynb`. It is based on the supplied `notebook757bdcf5cd (3).ipynb`, pins the inspected feature commits, and embeds the source update. The original downloaded notebook is preserved. No GitHub push is required to run the revised notebook.

## Confirmed faults

- `imageio_ffmpeg.read_frames` returns packed RGB bytes. Passing them to `Image.fromarray` broke every handoff mode. The extractor now uses `Image.frombytes` with decoded dimensions.
- `final` previously requested 0.05 seconds before EOF. It now selects the actual final decoded frame. `mid` counts decoded frames rather than trusting the nominal requested duration; a second sequential pass keeps memory bounded. `near_end` selects the default 0.5 seconds before the actual last frame. Each extracted image has an adjacent JSON record with frame index, decoded count, FPS and timestamp.
- The pipeline previously extracted one producer-selected frame and ignored the consumer's `handoff_policy`. Different consumers can now request different states from the same source clip.
- Existing MP4 files were reused after prompt, duration and reference edits. Resume now checks a generation signature. Changed or unverified old clips are renamed to `.previous-<timestamp>.mp4` before replacement; matching clips are reused without starting Wan.
- The supplied notebook's generation cell contains an uncommented fragment beginning `ython CUDA OOM`, which causes a SyntaxError before generation. Its composition cell also assumes output paths and card ordering that do not match the pipeline. The revised notebook fixes both.
- Mixed-resolution fallback clips are normalized before concatenation; matching streams retain the faster copy path.

## Science and pacing

The supplied notebook contains four chloride shots, with no litmus shot. The revised version preserves those four tests and adds a separately selectable paper-observation fixture. Its starting reference must already show a single tube, colourless liquid and damp red litmus paper held outside the liquid at the opening. Preparation and reagent addition are separate shots. The fixture does not invent a reagent quantity or claim to implement a complete ammonium analysis.

Structured positive prompts now lead with vessel preservation, target and observable location. Gas-test litmus instructions explicitly put the colour change on the paper and keep it outside the liquid. This is essential because the Lightning worker uses CFG=1 and does not evaluate the unconditional/negative branch; negative-prompt exclusions alone cannot constrain this path. See the [Lightning model card](https://huggingface.co/lightx2v/Wan2.2-Lightning) and the [RSC ammonia demonstration](https://edu.rsc.org/experiments/ammonia-fountain-demonstration/979.article) for the model setup and indicator principle.

The default profile preserves 2 seconds (33 frames at 16 FPS), given the reported OOM for longer shots. A lower-area 3-second profile remains explicitly experimental. Each shot requests one deliberate operation followed by a visible result. Optional 0.5x presentation produces roughly 4.125 seconds from 33 frames without additional GPU inference. It duplicates frames and cannot repair an incorrect chemical action. Compare longer shots through `benchmark.py` before adopting them; see `BENCHMARK.md`.

The RAG builder flags two deliberate operations, including reagent addition plus holding indicator paper. The pipeline stops before GPU work for these flagged plans, requiring an explicit split. It no longer copies a whole experiment's estimated video duration into every individual shot; unspecified steps default to 2 seconds.

## Generation time changes

- Keep the existing pre-encoding of prompt contexts and T5 release. CFG=1 now skips negative T5 encoding entirely. Base mode deduplicates shared negatives. Cached text is checked against its actual content so changed prompts cannot reuse stale embeddings.
- INT8 Linear now expands weights directly into the activation dtype and scales in place, removing the full-sized FP32 intermediate weight allocations from every forward. CPU numerical checks match the previous calculation exactly for FP32, FP16 and BF16. This remains a weight-only implementation, not a fused INT8 matrix multiplication kernel.
- SDPA preserves FP16 inputs instead of unconditionally converting them to BF16. It respects sequence lengths and scaling, so padding is not silently attended to. Backend speed still depends on Kaggle's PyTorch build; see [PyTorch SDPA documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention).
- Remember a successful fallback area for the same image dimensions and requested duration. Subsequent matching clips avoid repeating the known failed larger-area attempt. A fatal OOM restarts the worker and replays prompt preparation.
- Optional `--expert-source-dir=/kaggle/input/...` reads the two original Comfy FP8 expert files directly from a mounted dataset. It avoids their repeated network downloads and working-disk copies; mounted originals are never deleted. Expert conversion and transfer still occur. Expected filenames are `wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors` and `wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors`.
- Optional `--sampling-mode=base --base-steps=20` exposes a slower non-Lightning, CFG-enabled comparison. The notebook's `SAMPLING_MODE` selects it. It deliberately omits the four-step LoRA; changing its step count or CFG arbitrarily would not be a controlled comparison. Compare a single shot at the same image, area, duration and seed before considering the extra runtime.

## Further experiments

Measure wall time and CUDA telemetry before choosing more changes. Repeated checkpoint loading/conversion remains substantial. A converted-expert disk cache could remove quantization work but needs capacity for both INT8 experts plus conversion scratch; a CPU-resident expert cache needs measured RAM headroom. Fused INT8 kernels and alternate inference runtimes deserve an A/B test on the actual T4 build. They should be compared for visual detail and memory as well as speed. Frame interpolation may improve apparent smoothness but can corrupt thin indicator paper or transient precipitates; it is not enabled here.

## Validation and limits

Local tests exercise real H.264 frame extraction, independent consumer handoffs, signature-based resume, mixed-resolution composition, fatal OOM restart/cache replay, area reuse, prompt cache invalidation, INT8 numerical equivalence and SDPA masking/scaling. Notebook cells compile without executing GPU code. Full Wan inference, actual speed gains and scientific visual fidelity still require a fresh two-T4 Kaggle run. Inspect vessel count, paper versus liquid colour, operation sequence, pacing and final-state handoffs before using generated footage instructionally.
