import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import rag_interface


class RagInterfaceTests(unittest.TestCase):
    def test_cache_hit_is_normalized_without_generating(self):
        record = {
            "experiment_name": "Stored experiment",
            "clips": [{"name": "clip_1", "prompt_bundle": {"motion_prompt": "mix"}}],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            knowledge_dir = Path(temp_dir)
            (knowledge_dir / "stored.json").write_text(json.dumps(record), encoding="utf-8")
            with patch.object(rag_interface, "KNOWLEDGE_DIR", knowledge_dir), patch.object(
                rag_interface, "generate_procedure_with_llm"
            ) as generate:
                result = rag_interface.lookup_or_generate("stored experiment")

        generate.assert_not_called()
        self.assertEqual(result["clips"][0]["duration_seconds"], 1.0)
        self.assertIn("procedure_steps", result)

    def test_generated_plan_preserves_rendering_and_handoff_fields(self):
        generated = {
            "experiment_name": "Salt test",
            "procedure_text": "Add reagent.",
            "observation_text": "A white precipitate forms.",
            "procedure_steps": [
                {
                    "step_id": 1,
                    "instruction": "Add reagent.",
                    "observation": "A white precipitate forms.",
                    "state_before": "Clear solution",
                    "action": "Add reagent slowly",
                    "state_after": "White precipitate",
                    "reference_policy": "previous_state_keyframe",
                    "reference_clip": "clip_0",
                    "handoff_policy": "near_end",
                    "handoff_entity": "precipitate",
                    "duration_seconds": 2.5,
                }
            ],
        }
        with patch.object(rag_interface, "lookup", return_value=None), patch.object(
            rag_interface, "generate_procedure_with_llm", return_value=generated
        ):
            result = rag_interface.lookup_or_generate("a salt test")

        clip = result["clips"][0]
        self.assertEqual(clip["procedure_text"], "Add reagent.")
        self.assertEqual(clip["observation_text"], "A white precipitate forms.")
        self.assertEqual(clip["state_before"], "Clear solution")
        self.assertEqual(clip["action"], "Add reagent slowly")
        self.assertEqual(clip["state_after"], "White precipitate")
        self.assertEqual(clip["reference_clip"], "clip_0")
        self.assertEqual(clip["handoff_policy"], "near_end")
        self.assertEqual(clip["duration_seconds"], 2.5)
        self.assertEqual(clip["prompt_bundle"]["clip_duration_seconds"], 2.5)

    def test_invalid_plan_is_rejected(self):
        with self.assertRaises(ValueError):
            rag_interface.normalize_experiment_plan({"experiment_name": "Missing steps"})


if __name__ == "__main__":
    unittest.main()
