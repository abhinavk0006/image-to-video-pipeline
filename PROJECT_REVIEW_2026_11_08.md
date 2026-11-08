# Project review — 8 November 2026

## GitHub baseline inspected

| Repository | Active branch | Latest inspected commit |
|---|---|---|
| image-to-video-pipeline | feature/persistent-wan-worker | 16a7207 (22 October) |
| Wan2.2 | feature/persistent-wan-worker | 86eef84 (22 October) |
| Edu-video-gen-dataset | main and feature/persistent-wan-worker | 7522c9b (22 October) |

No later active-branch commits were found in the fresh GitHub fetch. The former
research branches are included in the persistent-worker history. Separate repos
still own dataset, orchestration and model runtime; consolidation does not mean
combining them into one Git repository.

## What is established

| Area | Evidence | Status |
|---|---|---|
| Styled chemistry cards | User reported cards look good | Visually confirmed by user |
| Old four-clip run | User supplied 4 decodable clips, 8.25 generated seconds, 3286.66 seconds wall time | Technical completion of that old run only |
| Chemistry correctness | User reported white instead of clear liquid, excessive doses, resets and partial-column clearing | Unresolved |
| Source quantities | 100 records audited; 23 chemistry records have 115 protocol variants; schema and adapter checks passed in October | Data/CPU validation, not video accuracy |
| Remaining records | 75 not covered, one cross-reference-only, one related-test-only; all have explicit draft contracts | Missing numbers remain unset |
| Physics | 58 records; deterministic circuit tooling exists, example DC smoke check passed | Manual values and textbook validation pending |
| Frame handoffs | Mid/near-end/final extraction, dependency and resume CPU checks passed | Generated chemical state at selected frame remains unverified |
| Material continuity | Same-sample tracks, cumulative budgets, overflow/reset checks and tool-contact constraints implemented | CPU validation only |
| Scientific evidence | Draft specifications, visibility annotations, reviewed handoffs, evidence timeline and bounded repair implemented | Human review required; no calibrated automatic verifier |
| Runtime optimizations | Bounded expert/prompt caching; optional conversion/VAE streaming; precision/placement experiments implemented | No measured speedup or longer-clip fit established |

The old run took about 54.8 minutes for 8.25 generated seconds. Expert loading
was about 1823 seconds, so loading/reuse is a plausible optimization target.
Those times are nested in denoising and must not be added twice. They do not
measure the new caching implementation. Playback slowdown is not longer native
generation. Passing tests or pushing a commit does not confirm end-to-end success.

## Prioritized improvements

1. **Run readiness and provenance (this change):** resolve an explicit starting
   image path, discover only the expected stem, reject ambiguous/corrupt/tiny
   references, report image and material failures together before GPU launch,
   save image hash/dimensions plus unresolved material details. Never certify
   image volume or substitute a dry-salt image for a prepared aliquot.
2. **Obtain source-complete dose plans:** resolve actual reagent concentrations,
   doses/endpoints and vessel working capacity. Preserve uncertainty; do not fill
   every missing amount with the same number. Strict mode stays blocked for the
   current chloride source gaps. Warn mode is an unvalidated visual experiment.
3. **Measure one baseline and one cache comparison:** same prepared reference,
   source plan, seed, precision, 33 frames and area. Separate cold/warm startup,
   expert cache misses/hits, full wall time, host/GPU peaks and OOMs. Inspect full
   clips for science and continuity before accepting runtime improvements.
4. **Then try duration/memory ablations:** compare VAE streaming separately from
   direct-to-GPU conversion; try 49 frames only after the 33-frame baseline fits.
   Do not enable multiple precision/placement changes simultaneously. Neither
   output streaming nor caching removes denoising activation memory.
5. **Calibrate scientific review:** independent reviewers, held-out experiment
   families, wrong-target/duplicate-vessel/omitted-action challenges and paired
   causal conditions. Report false acceptance and coverage alongside visual
   quality, realism and temporal consistency. Benchmarks need real generated
   footage, baselines, multiple seeds and confidence intervals.
6. **Physics enrichment:** wait for the promised manual, validate topology and
   NCERT layouts, then compare deterministic diagrams/animations as a separately
   labelled mode. No generic circuit values should be invented from titles.

## Current notebook

Use `Wan22_Kaggle_Continuous_State_Styled_Cards.ipynb`. Default chloride starts
with a prepared 1 mL aliquot and source steps 2–4. `INITIAL_IMAGE_PATH` accepts
an exact mounted/uploaded path; otherwise the expected stem is searched in
`/kaggle/input` and `/kaggle/working`. Matching JPG/PNG/WebP is allowed, while
multiple matches require explicit selection. `run_preflight.json` retains the
readiness report. Reference content still needs human inspection.

## Additional maintenance in this update

Benchmark probing now uses the shared explicitly owned FFmpeg decoder, including
its error cleanup. This addresses the pipe ResourceWarnings found in the CPU
suite; no GPU speedup is attributed to this housekeeping change.
