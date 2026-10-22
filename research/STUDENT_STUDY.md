# Draft learning evaluation protocol

Status: protocol/tooling only. No participants or outcomes have been collected.

## Research question

Do the generated demonstrations improve procedural understanding while avoiding
new misconceptions, compared with text plus static diagrams and suitable human
demonstration videos?

## Preparation

1. Choose reviewed experiments with matched learning objectives and prerequisites.
2. Have subject experts validate all content and quiz answers before exposure.
3. Define primary outcomes before collecting data: procedural understanding and
   misconception rate. Define delayed retention as a separate outcome.
4. Match information, study time, narration and quiz difficulty across conditions
   as closely as practical. Record remaining differences explicitly.
5. Obtain appropriate institutional/school approval and informed consent; follow
   guardian-consent requirements where applicable. Use pseudonymous identifiers.
6. Decide sample size using a justified power analysis. A convenient small pilot
   can identify usability problems but cannot establish effectiveness.

## Assignment and delivery

Input to `research.py study-assign` is a private JSON list:

```json
[{"pseudonymous_id":"participant_001","consent_recorded":true}]
```

Use `--participants participants.json --output-dir study/private --seed 42`.
The tool randomizes and balances assignment across `text_diagrams`,
`generated_video`, and `human_demonstration`. Keep allocation private. Plan
stratification or counterbalancing for prior knowledge and experiment difficulty;
the simple allocator does not implement that design automatically.

Give a pre-test, the assigned material, an immediate post-test and a prespecified
delayed test. Use parallel questions to reduce answer memorization. Assess:

- Identifying the correct next procedure step.
- Explaining why an action produces an observation.
- Detecting an incorrect apparatus connection or observation location.
- Transferring the principle to a new example.
- Explicitly rejecting common misconceptions.

Blinded assessors should score responses using a reviewed rubric. Record missing
follow-ups and exclusions; do not silently drop participants who performed poorly.

## Descriptive analysis

Use the CSV header:

```csv
pseudonymous_id,group,maximum_score,pre,post,retention
```

Run `research.py study-scores --scores scores.csv --output descriptive.json`.
The tool reports available counts, normalized score means and paired pre/post
percentage-point gains. Missing scores remain missing. It does not establish
causality, statistical significance, calibrated effect sizes or a treatment
benefit. A final analysis must account for assignment, prior knowledge, missingness,
clustering, multiple outcomes and the prespecified primary hypothesis.

## Pending

- [ ] Expert-approved materials and quiz/rubric.
- [ ] Ethics/consent and recruitment.
- [ ] Sample-size/design justification.
- [ ] Pilot and final collection.
- [ ] Delayed retention collection.
- [ ] Independent scoring and statistical analysis.
