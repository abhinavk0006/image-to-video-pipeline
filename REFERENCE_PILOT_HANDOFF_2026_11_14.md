# Reference pilot update

Use `Wan22_Kaggle_Continuous_State_Styled_Cards.ipynb` on
`feature/persistent-wan-worker` for the original pipeline.

The dataset now provides 33 draft configurations, 24 distinct reference images
and 15 chemistry records with image pilots. This is a pilot library, not image
coverage of all 100 records. All remain pending independent scientific review
and actual target-GPU generation.

New configurations:

- `aluminium_sol_delivery_excerpt`: clear receiving water before initial stock
  drops, then stirring; the source predicts a white hydroxide sol.
- `ferric_sol_delivery_excerpt`: reuses that same before-addition water scene
  in a fresh sample track; the source predicts a brown hydroxide sol.
- `neutralisation_first_transfer_excerpt`: reuses the existing clear-liquid
  calorimeter starting geometry for a fresh acid charge, with a short initial
  base-transfer excerpt and stirring. No numerical temperature is invented.

The new image was made with the built-in image-generation tool. It is saved as
`assets/reference_pilots/aluminium_sol_delivery_excerpt.png` in the dataset;
the exact prompt is saved under that pilot ID in
`assets/reference_pilots/generation_prompts.json`. Source snapshots, hashes,
scope, image identities and reuse decisions are in the manifest and catalog.

The stock/base quantities in the source describe the full procedure. These
two-second first-portion excerpts deliberately leave their own delivered dose
unset; they do not pretend to transfer the full charge instantly. Their material
plans therefore report `needs_source_detail`. Review scope and material warnings
before running; a draft preflight is not scientific acceptance.

Verification: all 33 source/image-bound pilots load and pass draft preflight;
26 relevant pipeline tests and three dataset quantity-contract tests pass.
The notebook generator compiles every Python cell. No A14B generation or research
accuracy score was produced on this laptop.

The separate motion-control experiment is on
`feature/fun-control-motion-guides`, with notebook
`Wan22_Fun_Control_A14B_Motion_Test.ipynb` and its own handoff. Its 24 wireframe
motions are review prototypes. GPU fit, speed, image quality and scientific
correctness still require Kaggle validation.
