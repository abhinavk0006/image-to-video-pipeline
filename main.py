"""CLI entrypoint for the educational image-to-video pipeline."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from config import PipelineConfig
from edu_video_rag_adapter import retrieve_pipeline_experiment
from knowledge_loader import load_experiment_from_json
from pipeline import ExperimentPipeline, ManualImageRequiredError, PipelineError
from rag_interface import lookup_or_generate
from video_generator import LocalVideoGenerator


def _slugify(value: str) -> str:
    text = value.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_") or "experiment"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the clip-by-clip educational image-to-video pipeline."
    )
    parser.add_argument(
        "--knowledge-file",
        default=None,
        help="Path to the experiment knowledge JSON file. Optional if --query is provided.",
    )
    parser.add_argument(
        "--query",
        default=None,
        help="Natural-language experiment query. Use --rag-repo-dir to retrieve grounded procedures from Edu-video-gen-dataset.",
    )
    parser.add_argument(
        "--experiment-name",
        default=None,
        help="Optional experiment output folder name. Defaults to a slug from knowledge JSON name.",
    )
    parser.add_argument(
        "--initial-image",
        default=None,
        help="Optional initial image file for the first clip.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Base output directory for generated clips and final video.",
    )
    parser.add_argument(
        "--knowledge-dir",
        default="knowledge",
        help="Knowledge directory label used in configuration.",
    )
    parser.add_argument(
        "--input-dir",
        default="inputs",
        help="Input directory label used in configuration.",
    )
    parser.add_argument(
        "--generator-executable",
        default="python",
        help="Command executable used to run the local Wan generation script.",
    )
    parser.add_argument(
        "--generator-script",
        required=True,
        help="Path to the local Wan generation script in Lightning AI.",
    )
    parser.add_argument(
        "--generator-working-dir",
        default=None,
        help="Optional working directory for generator execution.",
    )
    parser.add_argument(
        "--generator-arg",
        action="append",
        default=[],
        help="Extra argument for the local generator command. Repeat for multiple arguments.",
    )
    parser.add_argument(
        "--wan-repo-dir",
        default=None,
        help="Path to the local Wan 2.2 repository checkout. Used by the local wrapper.",
    )
    parser.add_argument(
        "--wan-model-preset",
        default="ti2v-5b",
        choices=["ti2v-5b", "i2v-a14b"],
        help="Wan model preset used by the local wrapper.",
    )
    parser.add_argument(
        "--rag-repo-dir",
        default=None,
        help="Path to the Edu-video-gen-dataset checkout. Enables its local TF-IDF RAG prompt builder for --query.",
    )
    parser.add_argument("--rag-top-k", type=int, default=1)
    parser.add_argument(
        "--rag-max-clips",
        type=int,
        default=None,
        help="Optional limit on retrieved procedure steps (use 1 for an end-to-end preview).",
    )
    parser.add_argument(
        "--rag-clip-duration-seconds",
        type=float,
        default=1.0,
        help="Duration assigned to each retrieved step; 1 second maps to the currently validated 17-frame Wan preview.",
    )
    parser.add_argument(
        "--continuity-mode",
        choices=["independent", "chain"],
        default="independent",
        help="Use verified references independently, or feed each clip's final frame into the next clip.",
    )
    parser.add_argument(
        "--wan-max-area",
        type=int,
        default=None,
        help="Requested Wan pixel area. CUDA OOMs can fall back to lower configured areas.",
    )
    parser.add_argument(
        "--wan-fallback-area",
        type=int,
        action="append",
        default=[],
        help="Lower Wan pixel area to try after CUDA OOM. Repeat in descending order.",
    )
    parser.add_argument(
        "--wan-memory-telemetry",
        action="store_true",
        help="Log per-GPU memory telemetry in the Kaggle Wan runner.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    knowledge_path = None

    if args.query:
        if args.rag_repo_dir:
            experiment_data = retrieve_pipeline_experiment(
                args.query,
                args.rag_repo_dir,
                top_k=args.rag_top_k,
                clip_duration_seconds=args.rag_clip_duration_seconds,
                max_clips=args.rag_max_clips,
            )
            rag_metadata = experiment_data.get("metadata", {})
            print(
                "RAG selected: "
                f"{experiment_data.get('name', 'unknown')} "
                f"(score={rag_metadata.get('retrieval_score', 0):.3f}, "
                f"source status={rag_metadata.get('source_status', 'unspecified')})"
            )
        else:
            experiment_data = lookup_or_generate(args.query)
        knowledge_dir = Path(args.output_dir).parent / args.knowledge_dir
        knowledge_dir.mkdir(parents=True, exist_ok=True)
        generated_name = str(
            experiment_data.get("name")
            or experiment_data.get("experiment_name")
            or "experiment"
        )
        experiment_name = args.experiment_name or _slugify(generated_name)
        knowledge_path = knowledge_dir / f"{experiment_name}.json"
        knowledge_path.write_text(json.dumps(experiment_data, indent=2), encoding="utf-8")
    else:
        if not args.knowledge_file:
            print("Provide either --knowledge-file or --query.", file=sys.stderr)
            return 1

        knowledge_path = Path(args.knowledge_file)
        if not knowledge_path.exists():
            print(f"Knowledge file not found: {knowledge_path}", file=sys.stderr)
            return 1

    experiment = load_experiment_from_json(str(knowledge_path))
    experiment_name = args.experiment_name or _slugify(experiment.name)

    config = PipelineConfig(
        base_output_dir=args.output_dir,
        knowledge_dir=args.knowledge_dir,
        input_dir=args.input_dir,
        experiment_name=experiment_name,
        continuity_mode=args.continuity_mode,
    )

    video_generator = LocalVideoGenerator(
        executable=args.generator_executable,
        script_path=args.generator_script,
        base_arguments=[
            *list(args.generator_arg),
            *([f"--wan-repo-dir={args.wan_repo_dir}"] if args.wan_repo_dir else []),
            f"--model-preset={args.wan_model_preset}",
            *([f"--max-area={args.wan_max_area}"] if args.wan_max_area else []),
            *(["--memory-telemetry"] if args.wan_memory_telemetry else []),
        ],
        working_directory=args.generator_working_dir,
        max_area=args.wan_max_area,
        fallback_areas=tuple(args.wan_fallback_area),
    )

    pipeline = ExperimentPipeline(config=config, video_generator=video_generator)

    try:
        final_video_path = pipeline.run(
            experiment,
            initial_image_path=args.initial_image,
        )
    except ManualImageRequiredError as error:
        print("Pipeline paused: manual image input is required.", file=sys.stderr)
        print(f"Clip: {error.clip_name}", file=sys.stderr)
        print(f"Image prompt file: {error.image_prompt_path}", file=sys.stderr)
        print(f"Place generated image at: {error.expected_image_path}", file=sys.stderr)
        print("After placing the image, rerun the same command.", file=sys.stderr)
        return 2
    except PipelineError as error:
        print(f"Pipeline failed: {error}", file=sys.stderr)
        return 1

    print(f"Pipeline completed successfully. Final video: {final_video_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
