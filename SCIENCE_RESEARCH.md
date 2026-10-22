# Scientific evidence branch

This branch adds optional scientific execution controls and research tooling.
It does not include a trained scientific verifier, completed GPU experiments,
expert-approved generated footage, or a conducted student study.

## Run the existing chemistry notebook

Use `Wan22_Kaggle_Science_Evidence_Optimized.ipynb` in a **fresh Kaggle session**.
Keep the current session running to preserve its outputs and baseline timings.
The notebook stays at 2 seconds and the existing spatial area, four Lightning
steps, per-tensor INT8 and balanced GPU placement. It enables bounded high-expert
CPU reuse, direct-to-GPU placement during expert conversion and CPU output
streaming from the VAE. Scientific mode defaults to
`observe`: it records review files without claiming that footage passed.

Generation performance and visual equivalence remain unmeasured until the new
notebook is run. The old 15â€“25 minute savings estimate is a hypothesis. CPU cache
eviction, available RAM, source acquisition and transfer costs determine savings.

## Scientific requirements and visibility

Each clip may declare `science_spec`, `visibility_targets`, `handoff_requirements`,
`render_mode`, `circuit` and `diagram_timeline`. The JSON loader and RAG adapter
preserve these fields. `source_reviewed` is false for automatically transcribed
contracts; set it true only after source verification, and supply `source_ids` in
clip metadata. For example:

```json
{
  "science_spec": {
    "source_reviewed": false,
    "entities": [{"id": "tube_1", "description": "the existing tube"}],
    "preconditions": ["The preceding preparation is complete"],
    "requirements": [
      {"id": "paper_position", "description": "Paper is held at the specified location", "critical": true, "evidence_kind": "interval"},
      {"id": "paper_result", "description": "The documented response occurs on the paper", "critical": true, "evidence_kind": "interval"}
    ],
    "temporal_order": [["paper_position", "paper_result"]]
  },
  "visibility_targets": [
    {"id": "paper", "bbox": [0.3, 0.1, 0.6, 0.3], "minimum_pixels": 24, "occluded": null}
  ]
}
```

The box above is **illustrative**; annotate the actual reference. Coordinates are
normalized `[left, top, right, bottom]`. A 24-pixel threshold is configurable and
not a scientific visibility guarantee. Occlusion is an annotation, not a hand
detector. Missing annotations are `unknown`. The report projects target size to
the requested Wan dimensions; inspect lower-area fallback clips separately.

CLI: `--science-mode off|observe|strict`. Strict mode requires passed visibility
and accepted scientific evidence; it pauses with review paths. Observe mode
allows footage to remain pending. A `review.json` records each requirement as
`pass`, `fail` or `unknown`, with reviewer, notes and timestamp intervals.
Nonzero intervals are required for temporal claims. `review.html` provides
clickable evidence times. Footage and contract hashes invalidate stale reviews.
Described intended outcomes never become observed facts automatically.

```bash
python research.py review --knowledge-file experiment.json --run-dir outputs/sequence
```

Source and generated-footage review are separate. Review temporal event windows
carefully: a later blue-paper endpoint cannot prove that the preceding action
was correct. Overlapping events should not be encoded as strictly ordered events.

## Reviewed handoffs and recovery

For an evidence-based handoff, set the consumer's `handoff_policy` to `reviewed`
and `handoff_requirements` to producer requirement IDs. In the producer review,
add candidates such as:

```json
{"timestamp_seconds": 1.5, "reviewer": "reviewer_1", "notes": "State is visible and unobstructed", "conditions": {"paper_result": "pass"}}
```

Put these in `handoff_candidates`. The producer must have an accepted review.
The selector picks the latest qualifying reviewed timestamp. It cannot infer
occluded chemistry, track objects automatically or approve unseen actions.

An optional `--science-verifier-script path.py` receives `--video`, `--contract`
and `--review`. It updates the review file using that same evidence format.
This branch supplies a **human-backed verifier interface**, not an accurate
pretrained semantic verifier. Model-based reviewers need independent calibration.

`--science-repair-budget 1` permits at most one extra generation per invocation
when the configured verifier explicitly rejects a clip. Unknown or stale reviews
never spend retry budget. Rejected footage and manifests are archived; a new
seed is recorded. A new invocation has a fresh budget. All attempts must count
toward research compute cost, not just the final accepted attempt.

```bash
python research.py repair-plan --knowledge-file experiment.json --failed-clip shot_2 --continuity-mode chain --output repair.json
```

Inspect the plan. Add `--run-dir outputs/sequence --apply` to archive affected
videos/manifests. The normal pipeline rerun regenerates those clips, preserves
independent outputs and invalidates dependent evidence by content hash.
Review the full affected sequence again. Cards and slow playback do not repair
scientific errors.

## Causal pairs and verifier challenges

```bash
python research.py causal-pair --knowledge-file experiment.json --pointer /clips/0/state_before --value-json '"A reviewed alternative starting condition"' --expected-json '{"control":"reviewed outcome A","intervention":"reviewed outcome B"}' --output-dir pairs/example
python research.py challenge --kind wrong_order --videos shot_1.mp4 shot_2.mp4 --output challenges/reversed.mp4
python research.py challenge --kind omitted_action --videos shot_1.mp4 shot_2.mp4 --output challenges/omitted.mp4
python research.py challenge --kind premature_observation --videos shot_1.mp4 --output challenges/premature.mp4
```

Paired inputs change one selected JSON value. Check reference images and all
unchanged descriptions for contradictions before running them. Expected outcomes
live in a separate draft manifest. Edited challenge footage is initially
**unlabelled**: reversing arbitrary clips or exchanging halves does not invariably
produce a scientific error. A reviewer must confirm the actual violation.
Duplicate-vessel and wrong-target challenges require reviewed source videos or
careful manual editing; those semantic edits are not automatically fabricated.

## Research metrics and blinded evaluation

The official VBench adapter runs six supported custom-input dimensions and saves
original raw outputs, commands, video hashes, package version and timing:

```bash
python research_metrics.py --videos shot_1.mp4 shot_2.mp4 --output-dir reports/vbench
```

This prepares commands. Install official VBench in a separate compatible GPU
environment and add `--execute` to run them after generation has stopped. The six
dimensions are subject/background consistency, motion smoothness, dynamic degree,
aesthetic quality and imaging quality. These custom-input results are not a
standard VBench leaderboard score or a percentage realistic. Static diagrams
and photorealistic footage must be evaluated in separate strata. Dynamic degree
is not an objective to maximize for a deliberately slow scientific procedure.
Source: https://github.com/Vchitect/VBench

`research.py blind --manifest videos.json --output-dir blinded --seed 42` copies
anonymized footage and creates separate public and private manifests. Input is
`{"videos":[{"path":"shot.mp4","method":"baseline"}]}`. Share only the reviewer
manifest and anonymized videos. Keep the key private.

After independent review, `research.py unblind --private-key private_key.json
--reviews reviewer_manifest.json --output unblinded.json` restores method labels
and reports per-method acceptance, coverage and mean human realism ratings on
their original 1–5 scale. Known challenge labels must have been independently
reviewed in the private input manifest to compute false-acceptance statistics.

`research.py summarize --records decisions.json --output scores.json` reports
acceptance and coverage, false acceptance on expert-labelled invalid examples,
accuracy on decided labelled cases, denominators and Wilson intervals. Each row
contains `decision`, and optional `ground_truth_reviewed` and `expected_valid`.
Pending cases remain visible. Aggregate by independent experiment; correlated
clips are not independent samples. Do not call clip acceptance full-sequence
success. `research.py review` separately records the joint sequence decision and requirement pass percentages
with evaluated/unknown coverage. Missing clips remain unconfirmed.

Use plain prompts, retrieval, contracts, reviewed handoffs and bounded recovery
as ablations with the same base model. Report both first-attempt and post-retry
accuracy. Include verifier time, all failed attempts and cache state in compute
comparisons. Keep held-out experiment families, multiple seeds and independent
expert review. Human realism ratings, scientific check percentages and VBench
scores answer different questions.

## Measured compute profiles

`research.py select-profile` selects from successful measurements with a
`measurement_id`, duration, area, wall time and per-GPU peaks. It never predicts
that unmeasured 3/4-second clips fit. Provide JSON records with keys `status`,
`measurement_id`, `seconds`, `area`, `wall_seconds`, `peak_allocated_bytes`, then
pass `--measurements profiles.json --seconds 3 --minimum-area 230400
--gpu-limit-gib 12 --gpu-limit-gib 12 --output selected.json`.
This is an offline selector, not an automatically integrated scientific scheduler.
Use actual reserved/driver memory and a safety margin when establishing limits.

## Physics references and schematic rendering

```bash
python circuit_renderer.py --circuit research/examples/ohms_law_circuit.json --output reference
python circuit_renderer.py --circuit research/examples/ohms_law_circuit.json --output circuit.mp4 --animate --seconds 6
```

SVG and PNG are rendered from explicit node positions, symbols and connections;
ideal DC nodal analysis controls meter readings. Labels identify the schematic
and simulated readings. This is an initial textbook-style backend, not a verified
reproduction of a particular NCERT figure or a complete physics simulator.
Check symbols and layout against the relevant textbook edition. Unsupported
components, floating nodes and inconsistent ideal sources fail explicitly.
No bulb-brightness model, AC, nonlinear diode model or physical apparatus view
is implied. A pipeline clip with `render_mode: circuit_diagram`, `circuit` and
`diagram_timeline` uses this renderer and bypasses Wan for that clip.

## Student learning study

See `research/STUDENT_STUDY.md`. Assignment and descriptive score tooling are
implemented. Participant recruitment, consent, expert quiz validation, delivery,
delayed follow-up and statistical analysis remain outstanding. No learning
benefit is claimed from producing assignment files.

## Rebuilding the notebook

With the three research repositories cloned into sibling folders, run
`python tools/build_science_notebook.py` from the pipeline checkout. It packages
explicit source files with hashes and writes the self-contained notebook. Rebuild
after runtime changes so the embedded payload matches the branch sources.
