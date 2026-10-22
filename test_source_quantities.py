"""Source facts must reach cards and budgets without conflating quantity roles."""
import unittest
from source_quantities import bind_source_quantities, quantity_label


class SourceQuantityTests(unittest.TestCase):
    def bind(self, facts, evidence=True):
        return bind_source_quantities({"clips": [{"action": "Add reagent", "metadata": {
            "quantities": facts,
            "source_evidence": [{"file": "manual.pdf", "pdf_page": 2, "experiment": "1"}] if evidence else [],
        }}]})["clips"][0]

    def fact(self, role="addition", qualifier="exact", value=1, unit="mL"):
        return {"reagent": "sample", "role": role, "qualifier": qualifier, "value": value, "unit": unit}

    def test_exact_addition_is_bound_with_citation(self):
        clip = self.bind([self.fact(value=2)])
        self.assertEqual(clip["liquid_state"]["dose"]["value"], 2)
        self.assertIn("manual.pdf PDF p2", clip["liquid_state"]["dose"]["source"])
        self.assertIn("sample: 2 mL [addition]", clip["procedure_text"])
        self.assertNotIn("before_ml", clip["liquid_state"])

    def test_roles_do_not_become_reagent_doses(self):
        for role in ("aliquot", "stock_preparation", "final_solution_volume", "vessel_capacity", "condition"):
            self.assertNotIn("dose", self.bind([self.fact(role=role)])["liquid_state"])

    def test_ranges_unknowns_multiple_additions_and_unsourced_values_stay_unresolved(self):
        for facts in ([{**self.fact(qualifier="range"), "minimum": 1, "maximum": 2}], [self.fact(qualifier="not_specified")], [self.fact(), self.fact()]):
            self.assertNotIn("dose", self.bind(facts)["liquid_state"])
        self.assertNotIn("dose", self.bind([self.fact()], evidence=False)["liquid_state"])

    def test_drops_are_not_converted_to_ml(self):
        state = self.bind([self.fact(value=3, unit="drops")])["liquid_state"]
        self.assertEqual(state["dose"]["count"], 3)
        self.assertNotIn("drop_volume_ml", state["dose"])

    def test_note_and_concentration_survive(self):
        label = quantity_label({**self.fact(qualifier="not_specified"), "concentration": "dilute", "note": "Few drops; count not given."})
        self.assertIn("Few drops", label)
        self.assertIn("dilute", label)


if __name__ == "__main__":
    unittest.main()
