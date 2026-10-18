# Science evidence and efficient generation

Branch: `codex/science-evidence-and-runtime-cache`, based on the persistent-worker feature work.
Implementation is separate from scientific acceptance and performance measurement.

## Runtime implementation

- [x] Reuse converted experts with bounded CPU memory and optional disk cache.
- [x] Reuse prompt embeddings across unchanged reruns.
- [x] Stream converted expert tensors directly to planned GPUs to reduce CPU overlap.
- [x] Stream completed VAE chunks to CPU without changing decoder precision.
- [x] Offer per-channel INT8 and explicit FP16 layer exemptions for controlled comparisons.
- [x] Offer contiguous GPU placement as an experimental alternative.
- [x] Wire options into wrapper and self-contained notebook; preserve 2-second baseline.
- [ ] Measure cache hits, stage times, OOMs and host-memory behaviour on two T4s.

## Scientific architecture

- [x] Represent intended conditions, invariants, temporal requirements and source provenance.
- [x] Keep observed evidence separate; never accept a condition because the prompt states it.
- [x] Add visibility preflight using annotated target regions and explicit uncertainty.
- [x] Add timestamped review evidence and a clickable local review report.
- [x] Select handoffs using reviewed evidence for the next action's preconditions.
- [x] Track dependencies and bounded repair/invalidation.
- [x] Implement an offline selector restricted to measured profiles and detail requirements; automatic scheduler integration remains pending.
- [ ] Validate on held-out experiment families with multiple seeds and expert review.

## Research evaluation

- [x] Prepare causal pairs by changing one reviewed initial condition or procedural variable.
- [x] Implement wrong-order, omitted-action and premature-observation challenge-video editing; labels require human review.
- [ ] Include reviewed duplicate-apparatus and wrong-target examples.
- [x] Report false acceptance, coverage, accuracy percentages, confidence intervals and failures.
- [x] Integrate established visual metrics separately from scientific review.
- [ ] Perform blinded comparisons and matched-budget ablations.

## Physics and education

- [x] Implement SVG/PNG circuit rendering with topology checks; textbook-figure validation remains pending.
- [x] Implement a separately labelled ideal-DC schematic animation that bypasses Wan.
- [x] Prepare a student study protocol, blinded assignment and score analysis.
- [ ] Recruit participants with appropriate consent and conduct the study.
- [ ] Measure understanding, misconceptions and delayed retention; publish no invented results.

## Explicit implementation limits

- [x] Add an adapter to run the official six VBench custom-input dimensions.
- [x] Add a sequential FFmpeg reader that explicitly closes extraction pipes.
- [x] Add activation capture and local precision-error comparison tooling.
- [ ] Run new regression tests and the optimized notebook on target hardware.
- [ ] Measure scientific correctness, realism, temporal quality and full-sequence success.
- [ ] Calibrate an actual semantic verifier against independent experts; no trained verifier is supplied.
- [ ] Add automatic object tracking and occlusion detection if human-backed review proves too slow.
- [ ] Integrate an automatic scientific compute scheduler after profiling data exists.
- [ ] Validate circuit layouts and symbols against chosen NCERT figures.
- [ ] Conduct full precision sweeps and verify candidate policies using generated videos.

Checked boxes mean implementation exists, not that target-hardware or scientific
validation has passed. The current branch was prepared without running new tests
or GPU inference; comparison and validation are the next session's work.
