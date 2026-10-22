"""Hash-bound human material review. No automatic chemistry or vision inference."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import time

CHECKS = ("appearance_matches_source", "dose_and_level_correct", "no_reset_or_overflow",
          "reaction_extent_correct", "tool_contact_correct", "action_order_correct")


def _video_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def material_review_status(video, clip):
    """Create pending evidence, invalidate stale reviews and return pass/fail/pending."""
    video = Path(video)
    path = video.with_suffix(".material-review.json")
    contract = {"state_before": clip.state_before, "action": clip.action,
                "state_after": clip.state_after, "liquid_state": clip.liquid_state,
                "tool_contact_policy": clip.tool_contact_policy}
    binding = {"video_sha256": _video_hash(video),
               "contract_sha256": hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()}
    try:
        review = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    except (ValueError, OSError):
        review = {}
    if not isinstance(review, dict) or review.get("binding") != binding:
        if path.exists():
            path.rename(path.with_name(path.name + f".previous-{time.time_ns()}"))
        review = {"binding": binding, "clip": clip.name, "reviewer": "", "source_reference": "",
                  "expected_contract": contract,
                  "checks": {key: {"verdict": "unknown", "intervals": [], "notes": ""} for key in CHECKS}}
        path.write_text(json.dumps(review, indent=2), encoding="utf-8")
    checks = review.get("checks", {})
    if not isinstance(checks, dict):
        return "pending", path
    if any(isinstance(checks.get(key), dict) and checks[key].get("verdict") == "fail" for key in CHECKS):
        return "fail", path
    if any(not isinstance(review.get(key), str) or not review[key].strip() for key in ("reviewer", "source_reference")):
        return "pending", path
    for key in CHECKS:
        item = checks.get(key, {})
        if not isinstance(item, dict) or item.get("verdict") != "pass":
            return "pending", path
        intervals = item.get("intervals", [])
        if not isinstance(intervals, list) or not intervals:
            return "pending", path
        for interval in intervals:
            if (not isinstance(interval, list) or len(interval) != 2
                    or any(isinstance(t, bool) or not isinstance(t, (float, int)) or not math.isfinite(t) for t in interval)
                    or not 0 <= interval[0] < interval[1]):
                return "pending", path
    return "pass", path
