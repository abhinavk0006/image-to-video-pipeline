import unittest
from models import Clip, Experiment, Scene, PromptBundle
from material_state import prepare_state_tracks, tool_constraint
from material_plan import compile_material_plan


def clip(name, **kwargs):
    return Clip(name=name, image_state=Scene("", "", ""), motion_prompt=PromptBundle(), **kwargs)


class MaterialContractTests(unittest.TestCase):
    def test_track_uses_latest_state_and_rejects_reset(self):
        a = clip("a", state_track="sample", liquid_state={"after_ml": 2})
        b = clip("b", state_track="sample", liquid_state={"before_ml": 2})
        prepare_state_tracks(Experiment(name="test", clips=[a, b]))
        self.assertEqual(b.reference_clip, "a")
        self.assertEqual(b.handoff_policy, "final")
        b.liquid_state["before_ml"] = 1
        with self.assertRaises(ValueError):
            prepare_state_tracks(Experiment(name="test", clips=[a, b]))

    def test_budget_accumulates_and_rejects_overflow(self):
        a = clip("a", action="Add solution", state_track="sample", liquid_state={
            "before_ml": 1, "dose": {"kind": "volume", "value": 2},
            "vessel_capacity_ml": 10, "safe_working_volume_ml": 5})
        b = clip("b", action="Add solution", state_track="sample", liquid_state={"dose": {"kind": "volume", "value": 1}})
        experiment = Experiment(name="test", clips=[a, b])
        prepare_state_tracks(experiment)
        report = compile_material_plan(experiment)
        self.assertEqual(report["clips"][1]["expected_after_ml"], 4)
        b.liquid_state["dose"]["value"] = 4
        with self.assertRaises(ValueError):
            compile_material_plan(experiment)

    def test_dispensing_tip_stays_outside_but_aspiration_is_not_forbidden(self):
        a = clip("a", action="Add reagent dropwise using a dropper")
        self.assertIn("above the vessel opening", tool_constraint(a))
        b = clip("b", action="Aspirate liquid using a dropper")
        self.assertEqual(tool_constraint(b), "")

    def test_unknown_acidification_does_not_become_zero_addition(self):
        a = clip("a", action="Acidify the aliquot", liquid_state={"before_ml": 1},
                 metadata={"quantities": [{"role": "addition", "qualifier": "not_specified", "reagent": "acid"}]})
        report = compile_material_plan(Experiment(name="test", clips=[a]))
        self.assertIsNone(report["clips"][0]["expected_after_ml"])
        self.assertEqual(report["status"], "needs_source_detail")


if __name__ == "__main__":
    unittest.main()
