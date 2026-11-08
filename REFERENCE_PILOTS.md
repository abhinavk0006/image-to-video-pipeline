# Run the source-linked image pilots

Open `Wan22_Kaggle_Continuous_State_Styled_Cards.ipynb` from
`feature/persistent-wan-worker` in a fresh Kaggle session with Internet and two T4 GPUs.
The notebook pins complete source snapshots and loads each reference image from the
dataset clone automatically. No separate reference upload is required.

Set `PILOT_ID` in the first configuration cell. The setup cell lists all available
IDs. The [dataset catalog](https://github.com/abhinavk0006/Edu-video-gen-dataset/blob/main/assets/reference_pilots/CATALOG.md)
contains every image, source protocol, clip count and reuse rationale.

Start with `carboxyl_litmus`, `salt_water_delivery` and `starch_iodine`, then test
`chloride_confirmation` and `sulphate_confirmation` for same-sample handoffs.
The 1/2/4 mL thiosulphate pilots are independent source variants sharing a compatible
pre-acid raster. Their sample tracks and dose budgets remain separate.

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
`assets/reference_pilots/` directory. Only records linked in the catalog have these pilot references; remaining
chemistry records and the source-pending physics records are not image-ready.

Every generation attempt uses the existing `run_benchmarked` capture. Check
`research_runs/` for technical reports and VBench input/status. VBench scores require
its evaluator and dependencies; prepared inputs alone are not scored results.
Scientific accuracy remains pending human review. Keep raw generated clips for
paper evaluation; slowed playback and cards do not count as generated video.

CPU loading, source/asset identity and continuity checks are covered by
`python -m unittest test_reference_pilots`. The expanded batch passed 45 pipeline tests and 5 dataset tests on CPU.
Real GPU quality and throughput still require Kaggle runs.
