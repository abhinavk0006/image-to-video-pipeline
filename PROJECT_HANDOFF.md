# WAN22 Educational Chemistry Video Pipeline

## Purpose

This project turns a natural-language chemistry-video request into a structured, educational video:

```text
User request
  -> experiment retrieval and planning
  -> deterministic procedure/observation cards
  -> Wan 2.2 image-to-video clips
  -> clip/card composition
  -> final instructional video
```

The current validation target is a one-test-tube ammonium-chloride qualitative salt-analysis experiment. The experiment is intentionally rendered as independent visual states wherever possible, with only the final continuation using a previous clip's state as a handoff.

## Repositories and responsibilities

### `Edu-video-gen-dataset`

This is the knowledge and data layer.

- Stores canonical chemistry experiments and metadata.
- Defines the experiment schema and validation rules.
- Provides TF-IDF retrieval through `scripts/query_rag.py`.
- Builds structured prompt bundles through `scripts/build_rag_prompt_bundle.py`.
- Emits procedure, observation, state, reference-policy, and rendering fields used by the pipeline.
- Can be queried locally on Kaggle without keeping an LLM resident.

The RAG layer selects the educational experiment. It does not control every visual detail of the current salt-analysis test; the notebook adds a controlled fixture so the generated clips use exactly one test tube and the intended state transitions.

### `image-to-video-pipeline`

This is the orchestration and presentation layer.

- `rag_interface.py` normalizes retrieved or Qwen-generated plans.
- `knowledge_loader.py`, `prompt_builder.py`, and `edu_video_rag_adapter.py` connect knowledge to structured prompts.
- `main.py` is the command-line entry point.
- `pipeline.py` coordinates planning, clip generation, and composition.
- `video_generator.py` talks to the persistent generator daemon and preserves clip/reference state.
- `kaggle_wan_wrapper.py` starts the Kaggle Wan worker.
- `stitcher.py` combines card and video segments with FFmpeg.
- The Kaggle notebooks exercise the full flow, including CPU card rendering and final composition.

This repository owns the end-to-end contract. It passes the selected knowledge record and controlled clip plan to Wan, then combines the generated clips with procedure, observation, title, and conclusion cards.

### `Wan2.2`

This is the GPU generation runtime.

- `wan/image2video.py` implements Wan image-to-video inference, T5/text encoding, VAE encoding/decoding, and expert execution.
- `quantization/kaggle_single_expert.py` loads streamed FP8 experts, merges the Lightning LoRA, quantizes linear layers, and dispatches transformer blocks.
- `quantization/kaggle_generate_video.py` owns the persistent JSON-lines daemon and staged expert lifecycle.
- The daemon keeps shared infrastructure alive across clip requests while swapping high- and low-noise experts when needed.
- Kaggle telemetry reports per-GPU allocation, reservation, free memory, and peak memory.

The runtime is designed for Kaggle's two Tesla T4 GPUs and is also usable in a local model-directory mode. VAE CPU offload is enabled for streamed Kaggle checkpoints, but remains opt-in/disabled for the local model-directory path.

## Current architecture

```text
Edu-video-gen-dataset
  chemistry_experiments.json
        |
        v
  TF-IDF retrieval / structured prompt bundle
        |
        v
image-to-video-pipeline
  RAG normalization + controlled fixture
  procedure/observation cards
  persistent daemon client
        |
        | JSON-lines requests: prompt, reference image, state/handoff metadata
        v
Wan2.2 persistent worker
  T5 text encoder (CPU-backed)
  VAE (GPU for encode/decode, CPU-offloaded during streamed denoising)
  high-noise and low-noise experts across GPU 0/GPU 1
        |
        v
  generated clip files
        |
        v
image-to-video-pipeline
  title -> procedure -> clip -> observation -> ...
  FFmpeg stitcher -> final educational video
```

The intended four-clip salt-analysis sequence is:

1. Dissolve dry salt in one test tube.
2. Acidify the clear solution.
3. Add silver nitrate and form a white silver-chloride precipitate.
4. Add ammonia to the same tube and dissolve the precipitate.

Clips 1–3 use explicit reference images for their target states. Clip 4 uses a chained handoff from clip 3 because the precipitate must remain visually continuous.

## Goals

### Product goals

- Generate scientifically understandable chemistry demonstrations from natural-language requests.
- Keep procedure and observation text deterministic and auditable.
- Preserve important visual state transitions without requiring every clip to be chained.
- Produce a final video containing both visual evidence and readable educational explanation.

### Engineering goals

- Maximize useful generation on limited Kaggle GPU memory.
- Keep Wan loaded persistently across multiple clips.
- Avoid resident Qwen and Wan competing for GPU memory.
- Use structured JSON contracts between retrieval, planning, generation, and composition.
- Make failures explicit, diagnosable, and recoverable.
- Keep the main branch stable while the persistent-worker work remains independently testable.

## Challenges and how they were handled

### Incorrect RAG retrieval

The broad chemistry query initially selected `Detect extra elements in an organic compound` with a score near `0.15`.

The notebook now uses the tested targeted query:

```python
query = "Detect one cation and one anion in a salt"
```

It also asserts that the selected title contains both `cation` and `anion`, so an unrelated retrieval cannot silently continue into generation.

### Generic plans did not guarantee visual continuity

An automatically retrieved multi-step plan could introduce extra vessels or ambiguous states. The end-to-end notebook therefore uses RAG for experiment identity and metadata, then creates a controlled four-clip one-tube fixture with:

- explicit `state_before`, `action`, and `state_after`;
- explicit procedure and observation text;
- negative prompts excluding extra tubes and apparatus;
- independent references for most clips;
- one deliberate chained handoff for the ammonia step.

### T5 and expert conversion memory pressure

The first persistent-worker attempt died while converting the low-noise expert. The worker now releases CPU T5 weights before converting the opposite expert and restores them afterward. This allowed low-noise conversion to complete on two T4 GPUs.

### Uneven two-GPU placement

Accelerate's previous placement could be order-biased and leave the inference-heavy GPU with too little headroom. The streamed Kaggle path now uses a capacity-normalized greedy mapper that explicitly places top-level modules and transformer blocks according to the configured GPU budgets.

The current notebook uses an intentionally asymmetric starting point:

```text
GPU 0 budget: 5 GiB
GPU 1 budget: 11 GiB
```

The mapper treats these as hard storage ceilings, not as a guarantee that inference activations will fit.

### VAE competing with denoising memory

The VAE stayed resident on CUDA during denoising and contributed to the inference OOM. For streamed checkpoints, the worker now encodes conditioning, moves the VAE to CPU during DiT denoising, then restores it to CUDA for decode. This trades transfer latency for activation headroom.

### Inference-time CUDA OOM

Expert conversion can succeed while the first denoising allocation still fails. The pipeline now retries only recognized CUDA OOM failures at strictly descending configured areas:

```text
345600 -> 230400 -> 172800
```

The same persistent daemon, references, and chain inputs are reused. Non-OOM failures remain fatal rather than being hidden by a fallback.

The fallback also handles a daemon that exits before returning a normal task response. Wan2.2 now reports Python-level CUDA OOM as a typed fatal response with `error_type="cuda_oom"` and exit code `75`, flushes the JSON response, and writes the final diagnostic to stderr. The pipeline drains and preserves daemon stderr, includes the exit code and diagnostic in its error, and restarts only the daemon for the next fallback area when the diagnostic explicitly identifies CUDA OOM. EOF, an unexplained nonzero exit, or an ordinary task error remains fatal; the pipeline never retries every daemon crash indiscriminately.

### Kaggle branch and commit confusion

The Kaggle setup originally fetched only `origin/main`, while the required fixes lived on feature branches. The notebook now fetches the correct ref for each repository before checking out the pinned commit:

```text
Wan2.2                  feature/persistent-wan-worker
image-to-video-pipeline feature/persistent-wan-worker
Edu-video-gen-dataset   feature/structured-reference-plans
```

## Current pinned revisions

The end-to-end notebook currently pins:

| Repository | Ref | Commit |
|---|---|---|
| `Wan2.2` | `feature/persistent-wan-worker` | `43be1f5` |
| `image-to-video-pipeline` | `feature/persistent-wan-worker` | `ebf1d33` |
| `Edu-video-gen-dataset` | `feature/structured-reference-plans` | `53b5222` |

The notebook setup correction and targeted RAG query are also on `image-to-video-pipeline`'s persistent branch:

```text
0a911c9  Fetch feature refs in Kaggle setup
c370118  Use targeted salt analysis RAG query
```

## Remaining challenges

- A complete four-clip run still needs successful validation on Kaggle after VAE offload and adaptive fallback.
- The quality and visual consistency of clips generated at `230400` or `172800` area must be compared with the native `345600` result.
- VAE CPU offload adds transfer time and may make short clips slower.
- Decode-time memory must be measured separately from denoising memory.
- The mapper's budget is a placement ceiling; attention, dequantization, CUDA allocator fragmentation, and activations can still exceed it.
- An OS-level `SIGKILL` cannot flush a typed response. If that happens, the parent can preserve only EOF, exit status, and any already-drained stderr, so the event remains an abnormal crash rather than an automatically assumed OOM.
- The notebook's card renderer is currently notebook-local Pillow code rather than a reusable presentation package or true LaTeX renderer.
- The current TF-IDF retriever is lexical. Semantic retrieval and stronger experiment/entity validation remain future improvements.
- Canonical dataset records still need broader observation-field cleanup and validation.
- The generated educational video needs visual review for chemistry correctness, vessel count, motion quality, and state handoff quality.
- The main branch does not automatically receive persistent-worker changes; the handoff document is duplicated there for discoverability, while feature work remains on the persistent branch until it is intentionally merged.

## Reproduction path on Kaggle

1. Start with a clean Kaggle session or remove only stale clones under `/kaggle/working`.
2. Run the notebook setup cell and confirm the three printed pinned commits.
3. Install the dataset requirements plus `ftfy`, `imageio-ffmpeg`, and `Pillow`.
4. Confirm the two required reference images are present under `/kaggle/input`.
5. Run the RAG cell and verify it selects the cation/anion salt experiment with a score around `0.63`, not the organic-compound result.
6. Render the deterministic cards before starting Wan.
7. Run the persistent generation command with:

```text
--wan-max-area 345600
--wan-fallback-area 230400
--wan-fallback-area 172800
--generator-arg=--gpu0-memory-gib=5
--generator-arg=--gpu1-memory-gib=11
--generator-arg=--memory-telemetry
```

8. Confirm logs show both expert conversions, VAE offload/restore, and any selected fallback area. If native inference OOMs, confirm the daemon reports `error_type=cuda_oom`/exit code `75` and the pipeline retries the next area.
9. Confirm four clips exist before running final composition.
10. Review the final stitched video and retain telemetry for future budget tuning.

## Branch policy

- `main` is the stable project reference and contains this handoff document.
- `feature/persistent-wan-worker` contains the active Kaggle persistent-worker implementation, adaptive OOM fallback, notebook pins, and this same handoff.
- Changes that alter the cross-repository contract should be documented here and validated in the end-to-end notebook before merging into `main`.
