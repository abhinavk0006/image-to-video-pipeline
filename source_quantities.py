"""Bind source quantity facts to prompts without inventing material budgets."""
from __future__ import annotations

import copy
import json
import math


def quantity_label(fact):
    label = fact["reagent"] + ": "
    qualifier = fact.get("qualifier", "not_specified")
    if qualifier == "not_specified":
        amount = "amount not specified"
    elif qualifier == "range":
        amount = f"{fact['minimum']:g}-{fact['maximum']:g} {fact.get('unit', '')}"
    else:
        value = fact.get("value", "")
        amount = f"{'about ' if qualifier == 'approximate' else ''}{value} {fact.get('unit', '')}"
    return (label + amount.strip()
            + (f" ({fact['concentration']})" if fact.get("concentration") else "")
            + f" [{fact.get('role', 'addition')}]"
            + (f" — {fact['note']}" if fact.get("note") else ""))


def bind_source_quantities(experiment):
    """Return an enriched plan; aliquots/stock/final volumes never become doses.

    Exact single additions can populate dose metadata. Starting volume, vessel
    capacity, drop calibration and chemistry outcomes always require their own
    evidence. Composite steps stay flagged for splitting.
    """
    result = copy.deepcopy(experiment)
    for clip in result.get("clips", []):
        metadata = clip.setdefault("metadata", {})
        facts = metadata.get("quantities", [])
        evidence = metadata.get("source_evidence", [])
        citation = "; ".join(f"{e['file']} PDF p{e['pdf_page']} ({e['experiment']})" for e in evidence)
        labels = [quantity_label(f) for f in facts]
        text = "; ".join(labels)
        if text:
            clip["quantity_constraint"] = text + ". Preserve roles: stock, aliquot, capacity and final volume are not interchangeable."
        clip["procedure_text"] = clip.get("action", "") + ("\n" + text if text else "")
        clip["observation_text"] = clip.get("state_after", "")
        state = clip.setdefault("liquid_state", {})
        scope = metadata.get("reaction_scope", "source_defined")
        if scope != "source_defined":
            state.setdefault("reaction_scope", scope)
        additions = [f for f in facts if f.get("role") in {"addition", "transfer"}]
        if "dose" not in state and len(additions) == 1 and citation:
            dose = additions[0]
            value = dose.get("value")
            if dose.get("qualifier") == "exact" and isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0:
                if dose.get("unit") == "mL":
                    state["dose"] = {"kind": "volume", "value": value, "unit": "mL", "source": citation}
                elif dose.get("unit") == "drops" and value > 0 and value == int(value):
                    state["dose"] = {"kind": "drops", "count": int(value), "source": citation}
        metadata["source_quantity_citation"] = citation
        metadata["source_quantity_roles_preserved"] = True
        # Preserve the complete evidence separately from encoder-limited prompts.
        metadata["quantity_facts_json"] = json.dumps(facts, ensure_ascii=False)
    return result
