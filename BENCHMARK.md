# Acceptance criteria and benchmark

The target is a long, scientifically correct sequence with reliable state handoffs and low wall time on two T4s. A longer or faster video is useful only when its observations and procedure remain correct. The selection metric is **scientifically accepted generated seconds per wall-clock second**, subject to complete sequence generation and acceptable visual detail. Playback slowdown and title cards do not count as generated seconds.

## Acceptance checks

- All declared clips and the stitched sequence decode successfully, with the expected frame counts and total duration.
- `mid`, `near_end` and `final` handoffs select the correct decoded indices. The images and JSON records are retained for inspection.
- A reviewer checks the source, apparatus identity, procedure order, observation location, observation correctness, pacing and dependent handoff states. Unreviewed footage stays `pending`, even if image similarity is high.
- Compare actual output resolution, peak allocated memory on each GPU, OOM outcomes, total generation wall time and per-stage timings.

## Fixed comparisons

`benchmark_matrix.json` defines a 2-second baseline, a 3-second lower-area candidate and a 4-second lower-area candidate. Longer candidates are experiments, not confirmed-fitting presets. Clip names, prompts and references stay the same, so deterministic seeds stay the same. The test overrides per-clip duration only for the controlled comparison.

Each case starts a fresh pipeline/worker process, persistent across its full sequence. It writes a separate output directory to prevent resume from contaminating timing. Downloads and OS caches are not reset. Use the same mounted weights and session conditions, repeat comparisons, and inspect the logs before attributing a speed difference to code. Expert-load timings are nested within denoising, not additive stages.

Run from the pipeline checkout on Kaggle after preparing the notebook's `experiment.json` and reference images:

```bash
python benchmark.py run \
  --knowledge-file /kaggle/working/wan_science_quality_chloride_safe_2s_lightning/experiment.json \
  --wan-repo-dir /kaggle/working/Wan2.2 \
  --output-dir /kaggle/working/benchmarks
```

The default runs only the baseline. Add `--cases longer_3s_lower_area` to select that case, or `--all-cases` to explicitly run the whole sweep. Optional `--expert-source-dir` uses mounted Comfy experts. There is no automatic area fallback during comparisons: an OOM is recorded as a failed configuration instead of silently testing a smaller resolution. The ordinary notebook retains its OOM fallbacks.

## Reports and scientific review

Each case saves its command, settings, logs, experiment copy, per-clip metadata, handoff images, `benchmark.json` and `scientific_review.json`. The run root contains environment/source hashes and CSV/JSON summaries. Missing stage measurements remain missing; they are never replaced with invented GPU results.

Fill `scientific_review.json` with actual boolean checks and reviewer/source notes. To refresh an existing report after review:

```bash
python benchmark.py collect \
  --knowledge-file RUN/CASE/experiment.json \
  --experiment-dir RUN/CASE/videos/sequence \
  --report-dir RUN/CASE/report \
  --pipeline-wall-seconds ORIGINAL_MEASURED_TIME
python benchmark.py summarize --run-dir RUN
```

Use the original measured time in `case.json`; omit the argument when it is unknown. Throughput remains null when timing or scientific approval is missing. The notebook also collects its existing run without starting another GPU job.

## Stage profiling

`--generator-arg=--profile-stages` records text-context handling, conditioning encode/offload, denoising including expert loading, decode preparation and VAE decode. It synchronizes both GPUs at phase boundaries. `*.metrics.json` records per-request peaks and preserves older attempts separately. Per-phase peak values are explicitly cumulative since the request began, not isolated phase peaks. Allocated/reserved values are PyTorch allocator measurements, not a complete measure of every driver's allocation.

Use the last completed stage and `incomplete_stages` to distinguish a denoising OOM from a decoding OOM. This evidence should decide whether to prioritize converted-expert caching, attention memory, VAE streaming or GPU placement. Sensitivity-aware quantization is still an unimplemented experiment; these benchmark additions do not make it active.

## Current defaults and limits

The notebook now defaults to 2-second/33-frame generation because longer clips previously OOMed. Its `safe_2s` name preserves that temporal baseline; it is not a promise that every spatial resolution fits. Optional 0.5x playback makes 33 frames last about 4.125 seconds without extra inference. The RAG per-step fallback is also 2 seconds. Longer shots remain opt-in and require the benchmark.

Local regression tests and synthetic-video composition validate the tooling. Actual T4 timing, OOM limits and scientific quality must come from the target hardware and human review. No successful longer-GPU-run result is claimed here.
