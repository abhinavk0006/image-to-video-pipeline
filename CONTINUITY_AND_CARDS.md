# Continuous material state and science cards

## Use this notebook

`Wan22_Kaggle_Continuous_State_Styled_Cards.ipynb` replaces the old quality notebook
for this iteration. Use a fresh Kaggle session with Internet and two T4s. It only
uses the selected dataset protocol and its actual source quantities. The default
chloride confirmation starts with a prepared 1 mL test aliquot and requires a
matching `chloride_1ml_aliquot.png` reference. Dry-salt preparation is a separate
protocol, and the old dry-salt image must not be used for this initial state.
The other operations consume actual generated final frames.
The output folder is different from the preceding experiment, preserving its clips.

The 33-frame, approximately 2.06-second generation workload and area fallbacks are
unchanged. The 1280×720 cards and final presentation are rendered on CPU; this does
not increase native video detail. Slow playback does not add generated frames.
Longer generation remains experimental given the reported GPU1 peak of 14.17 GiB.

### Source quantities and unresolved doses

The adapter accepts `protocol_selections={"chem_12_301": "chloride_silver"}`;
the CLI equivalent is `--rag-protocol chem_12_301=chloride_silver`. Alternative
tests are never concatenated. `source_quantities.py` carries typed quantity facts,
their roles, notes and PDF citations into plan metadata and procedure cards.
Only a single exact sourced addition/transfer becomes a numeric reagent dose.
An aliquot, stock quantity, vessel capacity, range or unspecified amount does not.
Drop counts remain drops without a sourced calibration.

The default source protocol has no exact HNO3, AgNO3 or NH4OH doses or vessel
working capacity. `MATERIAL_PLAN_MODE="strict"` therefore stops before GPU work
until those gaps are resolved. `warn` permits a visibly unvalidated trial and
records the unresolved budget. Neither mode establishes scientific accuracy.
Use source-backed `MATERIAL_OVERRIDES`; never guess equal reagent volumes.

The notebook pins complete Git snapshots of all three repositories, avoiding the
old partial source overlay that could leave the dataset and runtime inconsistent.
For other protocols, review `SAMPLE_TRACKS`, separate reference images and any
composite steps. No generic `sample_id="reaction"` is treated as proof that all
operations share a vessel. Physics numerical enrichment awaits its source manual.

Bounded high-expert CPU caching is available with host headroom checks. Disk
expert caching, VAE output streaming and direct-to-GPU expert conversion are
configurable in the notebook. Precision and placement experiments remain opt-in;
see Wan's `quantization/RUNTIME_OPTIMIZATIONS.md`. No speedup is established yet.

## General contracts

Declare a `state_track` for each physical sample/vessel sequence. A later operation
on that track automatically references its latest generated clip, including when
other tracks occur in between. Final-frame handoffs are the default for these
tracks. A declared consumer handoff policy can still select mid/near-end/offset.
Use that only when the required state exists at that time; the extractor does not
judge whether a reaction has completed.

Explicit generated-state references take priority over static input paths.
Use `starts_new_sample: true` plus a separate `input_frame_path` for an intentional
fresh aliquot/reset. Sample identity is never guessed from a generic vessel name.
An older-state reference on the same track is rejected rather than silently
resetting its contents. Existing untracked independent shots retain their mode.

Optional JSON fields, applicable to any experiment:

```json
{
  "state_track": "sample_A",
  "tool_contact_policy": "above_rim",
  "liquid_state": {
    "before_appearance": "copy the documented starting appearance",
    "after_appearance": "copy the documented ending appearance",
    "solid_change": "copy the documented solid behaviour",
    "level_change": "increase"
  }
}
```

Optional numeric `before_ml`, `added_ml`, `removed_ml`, `after_ml`, and
`max_fill_fraction` must come from the source. When all four volume quantities are
specified, their arithmetic balance is checked before starting the worker.
`max_fill_fraction` is a fraction of internal vessel height, not inferred mL.
No fixed dose, colour, reaction or universal fill percentage is invented. Additions
can raise the level; removal/evaporation should declare the appropriate change.

`tool_contact_policy` supports `auto`, `above_rim`, `above_liquid`,
`source_defined`, and `immersion_required`. Auto only adds the above-rim rule for
dropper delivery actions without an explicit sampling/immersion instruction.
Aspiration and procedures requiring immersion must keep their source-defined
contact. Positive prompts carry these rules because CFG=1 ignores negative prompts.

Each clip directory includes `reference_provenance.json`: parent clip, selected
handoff mode, reference SHA256, state track, expected liquid state, and an explicit
`scientific_state_verified: false`. Generation manifests also include provenance.
These are audit records of conditioning, not measurements of generated chemistry.

## Cards

`science_cards.render_cards` returns all pages in order. Use chemistry (blue/purple),
physics (cyan/blue), or biology (green/blue) themes. Title, procedure, observation and
conclusion layouts share dark backgrounds, large text and a gradient accent stripe.
Long text paginates instead of being clipped. Overlong headings/footers raise errors.

Pillow renders text; matplotlib renders `$...$` or `$$...$$` expressions as separate
math blocks. Use mathtext notation such as `\\mathrm{AgCl}` with a single backslash
in the actual formula. This is a LaTeX subset, not a full TeX or `mhchem` compiler;
unsupported commands raise an explicit error. Dependencies: Pillow>=10.1, matplotlib,
and numpy (also a matplotlib dependency). The Kaggle notebook installs these.

## Review checklist for the next GPU run

- [ ] Initial reference matches the intended apparatus, salt and quantity.
- [ ] Dissolution, precipitation and colour match the documented state at each step.
- [ ] Meniscus begins at the preceding generated level and changes only through the operation.
- [ ] No unexplained draining, refill, duplicate vessel or oversized addition.
- [ ] Delivery tip stays above the rim/liquid when required; no accidental immersion.
- [ ] Action and result are visible at readable speed; handoff state is suitable.
- [ ] Inspect `reference_provenance.json` and compare actual selected images.
- [ ] Review generated clips, not only the final frame.
- [ ] Complete the new appearance, quantity/continuity and tool-contact human checks
  in `benchmark_report/scientific_review.json`; use notes for nonapplicable checks.
- [ ] Reviews are bound to video/source hashes. Changed or previously unbound
  evidence archives the old review and resets its checks to pending.
- [ ] Compare accepted generated seconds per wall second and stage timings with the
  preceding 3286.657-second run. Do not add nested expert loads to denoising time.

The reported run had four decodable clips but four pending scientific reviews.
These changes repair conditioning selection and add constraints; Wan can still
violate prompts. No automatic colour, volume, contact or reaction verification is
claimed. A visually incorrect final frame will propagate its error unless rejected
or regenerated after review. The separate research branch contains additional
verification/runtime work and is not merged by this update.

## Rebuild

### Preview import recovery

If `from frame_extractor import FrameExtractor` reports `ModuleNotFoundError`
after successful generation, the notebook kernel cannot see the repository's
source directory. `subprocess.run(..., cwd=pipeline)` sets the child's working
directory, not the notebook kernel's import path. The exact reason an earlier
path entry was lost cannot be determined from the log alone.

Add this at the top of the failing preview cell and rerun only that cell:

```python
import sys, importlib
from pathlib import Path
pipeline = Path("/kaggle/working/image-to-video-pipeline")
if not (pipeline / "frame_extractor.py").is_file():
    raise FileNotFoundError(f"Restore pipeline sources at {pipeline}; keep existing videos")
sys.path.insert(0, str(pipeline))
importlib.invalidate_caches()
```

The revised notebook adds its own source-path bootstrap to each local-import cell,
including shot planning, cards, preview and composition. Preview also imports
Pillow and display explicitly. No new GPU generation is needed for this fix.

### Build from source

Run `python tools/build_continuity_notebook.py --wan-repo <Wan2.2 checkout>
--rag-repo <Edu-video-gen-dataset checkout> --output-dir <artifact directory>`.
The two dependency checkouts must match the pinned persistent/structured-plan bases;
the builder embeds selected files from them. Commit hashes and file SHA256s are
included in the notebook payload manifest.
