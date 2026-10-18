# Continuous material state and science cards

## Use this notebook

`Wan22_Kaggle_Continuous_State_Styled_Cards.ipynb` replaces the old quality notebook
for this iteration. Use a fresh Kaggle session with Internet and two T4s. It only
requires the initial `salt_analysis_one_test_tube_dry_salt.jpeg` reference for the
chloride sequence. The other operations consume actual generated final frames.
The output folder is different from the preceding experiment, preserving its clips.

The 33-frame, approximately 2.06-second generation workload and area fallbacks are
unchanged. The 1280×720 cards and final presentation are rendered on CPU; this does
not increase native video detail. Slow playback does not add generated frames.
Longer generation remains experimental given the reported GPU1 peak of 14.17 GiB.

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

Run `python tools/build_continuity_notebook.py --wan-repo <Wan2.2 checkout>
--rag-repo <Edu-video-gen-dataset checkout> --output-dir <artifact directory>`.
The two dependency checkouts must match the pinned persistent/structured-plan bases;
the builder embeds selected files from them. Commit hashes and file SHA256s are
included in the notebook payload manifest.
