# General liquid-dose and reaction controls

## Why the previous changes were insufficient

Final-frame chaining repaired a static-reference reset path, but a diffusion model
can still invent meniscus changes or split a reaction spatially inside a clip.
The chloride fixture also gave every reagent the same generic dropper action,
without specifying dose amounts or vessel capacity. An end-state sentence alone
cannot establish dose, completeness, mixing, timing or observed correctness.

## Catalogue audit, 22 October 2026

Inspection of `Edu-video-gen-dataset/data/chemistry_experiments.json` found 100
records: 42 chemistry and 58 physics. No procedure-step keys represented a
structured volume, dose, drop count or capacity. A search of step instruction,
action and quantity-constraint text for numeric mL/L/cm3/g/mg/drop quantities and
common written drop counts found one chemistry candidate: `chem_9_001`, "one drop"
in the ink-diffusion experiment. The other 41 chemistry records had no match in
those step fields. This is a limited text audit, not a proof that every possible
wording or other field lacks a quantity. Source records remain marked source_review.

The salt-analysis record `chem_12_301` describes generic preparation/group tests/
confirmatory tests. Its exact source pages and quantitative procedure have not been
transcribed. The notebook's four-shot fixture is separately assembled from that
retrieval and supplies no quantities either. No local copy of the referenced
`Manual_01.pdf` was found in the checked Downloads and Documents/Codex trees.

This update does not invent or bulk populate quantities. Fixing the catalogue
requires source transcription and review; improving prompts cannot recover facts
that the supplied knowledge does not contain.

## Source-defined dose schema

Per-clip `liquid_state` can include:

- `before_ml`, `added_ml`, `removed_ml`, `after_ml` (source-defined mL).
- `vessel_capacity_ml` and optional `safe_working_volume_ml` from the apparatus/
  procedure. Capacity carries across the sample track; a lower working limit takes
  precedence. No universal tube size or fill percentage is assumed.
- `dose`: one of the structures below. These are field templates, not suggested
  reagent amounts. Replace symbolic placeholders with actual JSON numbers/text.

```text
{"kind": "volume", "value": SOURCE_ML, "unit": "mL", "source": "manual/page/step"}
{"kind": "drops", "count": SOURCE_INTEGER, "source": "manual/page/step"}
{"kind": "endpoint", "endpoint": "documented endpoint",
 "max_added_ml": SOURCE_UPPER_BOUND_ML, "source": "manual/page/step"}
{"kind": "none"}
```

Drop-count instructions can include calibrated `drop_volume_ml` or a sourced
`max_added_ml`. There is no built-in drops-to-mL conversion. Drop count alone
controls the requested animation but cannot establish a cumulative volume budget.

The additive budget computes nominal before + added - removed across referenced
clips and rejects impossible negative totals, inconsistent declared endpoints,
capacity changes and capacity exceedances before GPU startup. Missing quantities
remain unknown; a predecessor's addition is never forgotten at a clip boundary.
Each dose belongs to its own step; equal clip duration does not imply equal doses.

Real mixtures can contract/expand, and dissolved solids can affect volume. The
default `volume_model: "additive_budget"` is a nominal planning model, not a claim
that physical liquid volumes always add exactly. Use `volume_model: "source_defined"`
with a documented `after_ml` when that assumption is inappropriate; provide suitable
source-defined bounds. The planning report remains explicitly unverified.

## Clarity and reaction extent

Use explicit fields, validated against the actual procedure:

- `after_clarity`: clear / cloudy / opaque / source_defined.
- `after_colour`: source-defined colour; clear does not mean colourless.
- `reaction_scope`: whole_contents / localized / interface / stratified /
  source_defined. Only a declared whole-content transformation receives the
  complete-column endpoint constraint.
- `after_solid: "none"` means complete disappearance is actually required by the
  source. It is not inferred for every salt or reagent addition.
- `mixing_instruction`: the source's mixing operation, if any. No automatic
  shaking/stirring is added. A separate operation may require another shot.
- `minimum_visible_seconds`: a source/review-defined minimum. A shorter request
  fails preflight; the code never silently increases the GPU workload.

The positive prompt distinguishes transformation of existing material from
displacement of an unchanged cloudy plug. It explicitly requests transparency
throughout a required clear endpoint and stops dispensing at the documented dose
or endpoint. Interface reactions and actual layered systems retain their own scope.

## Plan and review modes

`--material-plan-mode warn|strict` writes `material_plan.json` before generation.
Strict stops unresolved dose/capacity plans before preparing T5 or a GPU worker.
Warn allows an unvalidated visual trial with all gaps printed and recorded.
Contradictory numeric plans fail in both modes. The library/CLI defaults to warn
for compatibility; the revised notebook defaults to strict for this iteration.

In the notebook, fill `MATERIAL_OVERRIDES` as a mapping from actual clip names to
the reviewed `liquid_state` fields. Supply initial volume/capacity at the first
operation and each subsequent dose; known totals propagate automatically. Preserve
all source citations. The adapter also carries source-step liquid_state data when
the retrieved canonical record contains it, even if a prompt export omits it.

Each liquid clip gets `<clip>.material-review.json` bound to the video bytes and
its expected material contract. It starts with unknown checks. Source-linked human
review requires reviewer, source_reference and pass/fail/unknown verdicts for
appearance, dose/level, no reset/overflow, reaction extent, contact and action order.
Every pass needs nonzero evidence intervals such as `[[0.2, 1.8]]` in video seconds.
Review the actual interval and complete clip, including inappropriate late changes.
An accepted endpoint alone cannot prove correct actions throughout a clip.

`--material-review-mode observe|strict` controls this separate evidence gate:

- Observe records pending reviews and permits continuation; an explicit fail stops.
- Strict stops at pending/fail before accepting or chaining the saved clip.
- Complete the review, then resume. Matching accepted clips are reused, not
  regenerated. Pending/failed clips are not automatically repaired or accepted.
- Changed videos/contracts archive old reviews and require fresh evidence.

Strict review can add human time and require worker restarts between invocations.
It is optional and is separate from numeric preflight. No trained vision model,
automatic meniscus measurement, reaction detector or guaranteed visual repair is
claimed. These controls constrain intended generation and prevent known/unchecked
states from being propagated when strict review is selected.

## Next run

Fill the actual source amounts before using strict material planning. Keep the
existing two-second/area settings until longer configurations have hardware evidence.
Inspect raw footage and each actual handoff for clarity, complete reaction, tool
contact, dose, overflow and unexpected level changes. Review with multiple
experiment families, including clear coloured solutions and legitimate layered
systems, before claiming a general visual improvement. No new GPU validation or
regression tests were run as part of this implementation.
