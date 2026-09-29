"""Record and score physical trials for gates 2 and 4 (hardware; see docs/gate2-protocol.md, docs/gate4-protocol.md).

Trials are appended as JSON lines so raw outcomes and failure reasons are kept separately
from simulation results (PLAN.md: record raw trials and failure reasons separately).
Scoring applies the pre-registered thresholds and never edits the raw log.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

GATE2_SKILLS = ("shared_zone_pick", "close_packed_extraction", "insertion", "reorientation", "placement")
GATE4_ACTIONS = ("chow", "pung", "kong_exposed", "kong_concealed", "kong_added", "win_self_draw", "win_discard")
GATE2_SKILL_TRIALS, GATE2_SKILL_RATE = 50, 0.90
GATE2_RECOG_OVERALL, GATE2_RECOG_TYPE = 0.90, 0.80
GATE4_HANDS, GATE4_HANDS_PASS = 5, 3
GATE4_ACTION_TRIALS, GATE4_ACTION_PASS = 10, 8


def log_path(gate: int, root: str | Path = "reports") -> Path:
    return Path(root) / f"gate{gate}" / "hardware_trials.jsonl"


def record(gate: int, trial: dict, root: str | Path = "reports") -> dict:
    """Append one trial. Required: ``kind``; ``success`` (bool); ``reason`` when it failed."""
    if "kind" not in trial or "success" not in trial:
        raise ValueError("a trial needs 'kind' and 'success'")
    if not trial["success"] and not trial.get("reason"):
        raise ValueError("a failed trial needs a 'reason'")
    row = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "gate": gate, **trial}
    p = log_path(gate, root)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as f:
        f.write(json.dumps(row) + "\n")
    return row


def load(gate: int, root: str | Path = "reports") -> list[dict]:
    p = log_path(gate, root)
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()] if p.exists() else []


def _rate(rows):
    n = len(rows)
    k = sum(1 for r in rows if r["success"])
    return k, n


def score_gate2(rows: list[dict]) -> dict:
    """Skills: >=90% over the first 50 held-out trials each. Recognition: 10 held-out views per type."""
    out = {"skills": {}, "recognition": None}
    ok = True
    for skill in GATE2_SKILLS:
        trials = [r for r in rows if r["kind"] == skill and r.get("held_out", True)][:GATE2_SKILL_TRIALS]
        k, n = _rate(trials)
        passed = n == GATE2_SKILL_TRIALS and k >= GATE2_SKILL_RATE * n
        reasons: dict[str, int] = {}
        for r in trials:
            if not r["success"]:
                reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
        out["skills"][skill] = {"successes": k, "trials": n, "passed": passed, "failure_reasons": reasons}
        ok &= passed
    views = [r for r in rows if r["kind"] == "recognition"]
    if views:
        per_type: dict[int, list[bool]] = {}
        for r in views:
            per_type.setdefault(int(r["tile_type"]), []).append(bool(r["success"]))
        overall = sum(sum(v) for v in per_type.values()) / len(views)
        worst = min(sum(v) / len(v) for v in per_type.values())
        complete = len(per_type) == 34 and all(len(v) >= 10 for v in per_type.values())
        rp = complete and overall >= GATE2_RECOG_OVERALL and worst >= GATE2_RECOG_TYPE
        out["recognition"] = {"views": len(views), "types": len(per_type), "overall": overall,
                              "worst_type": worst, "complete": complete, "passed": rp}
        ok &= rp
    else:
        ok = False
    out["passed"] = ok
    return out


def score_gate4(rows: list[dict]) -> dict:
    """>=3 of 5 complete hands without corrective touch; >=8 of 10 trials per call form and win route."""
    hands = [r for r in rows if r["kind"] == "hand"][:GATE4_HANDS]
    good = [r for r in hands if r["success"] and not r.get("corrective_touch", False)]
    out = {"hands": {"complete_without_touch": len(good), "trials": len(hands),
                     "passed": len(hands) == GATE4_HANDS and len(good) >= GATE4_HANDS_PASS}, "actions": {}}
    ok = out["hands"]["passed"]
    for a in GATE4_ACTIONS:
        trials = [r for r in rows if r["kind"] == a][:GATE4_ACTION_TRIALS]
        k = sum(1 for r in trials if r["success"] and not r.get("corrective_touch", False))
        passed = len(trials) == GATE4_ACTION_TRIALS and k >= GATE4_ACTION_PASS
        out["actions"][a] = {"successes": k, "trials": len(trials), "passed": passed}
        ok &= passed
    out["passed"] = ok
    return out
