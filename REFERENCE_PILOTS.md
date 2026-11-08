# Run the source-linked image pilots

Open `Wan22_Kaggle_Continuous_State_Styled_Cards.ipynb` from
`feature/persistent-wan-worker` in a fresh Kaggle session with Internet and two T4 GPUs.
The notebook pins complete source snapshots and loads each reference image from the
dataset clone automatically. No separate reference upload is required.

Set `PILOT_ID` in the first configuration cell:

| ID | Trial | Generated shots |
|---|---|---|
| `chloride_confirmation` | Prepared clear aliquot; acidification, silver nitrate, ammonia | 3 |
| `thiosulphate_cross` | Prepared clear solution over a cross; delivery, swirl, observation | 3 |
| `starch_heating_excerpt` | Ongoing heating and stirring of a prepared starch dispersion | 1 |

Run one pilot at a time. Use a fresh kernel/configuration execution when changing
pilot IDs; output directories include the pilot ID. Keep `safe_2s` for the first run.
Inspect the starting image, material report and cards before the GPU generation cell.
Inspect raw clips and final-frame handoffs afterwards, including meniscus continuity,
whole-sample appearance and contact between tools and liquids.

These are draft visual excerpts, not validated full experiments. Exact chloride
reagent doses and vessel limits remain unresolved. The cross trial does not measure
reaction time, and the starch clip does not complete the source's ten-minute hold.
`MATERIAL_PLAN_MODE="warn"` enables visual trials with unresolved details retained;
use strict modes after supplying source-backed details and performing reviews.
Generated images do not establish exact mL, apparatus calibration or correctness.

Each pilot binds the image SHA-256 and all selected protocol-step hashes. Changed
assets or source steps require review and rebinding before generation. Exact image
generation prompts and source snapshots are in the dataset's
`assets/reference_pilots/` directory. Only three pilot records have linked references;
other chemistry records and the source-pending physics records are not image-ready.

Every generation attempt uses the existing `run_benchmarked` capture. Check
`research_runs/` for technical reports and VBench input/status. VBench scores require
its evaluator and dependencies; prepared inputs alone are not scored results.
Scientific accuracy remains pending human review. Keep raw generated clips for
paper evaluation; slowed playback and cards do not count as generated video.

CPU loading, source/asset identity and continuity checks are covered by
`python -m unittest test_reference_pilots`. Real GPU quality and throughput still
require Kaggle runs.
