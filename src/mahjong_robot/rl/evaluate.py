"""Gate 1 evaluation: paired, seat-balanced, held-out hands.

Every test hand is played once per robot-seat player (the policy, the legal random player,
the training heuristic) with the same wall seed and the same opponent seeds, so the hands
are paired. The robot seat is ``seed % 4`` and the test seeds start at a multiple of 4, so
each seat appears equally often. Opponents are the held-out heuristic, never used in
training or checkpoint selection.
"""

from __future__ import annotations

import csv
import json
import math
import multiprocessing as mp
from pathlib import Path

import torch

from ..game.engine import IllegalAction
from ..game.table import play_hand
from . import seeds as S
from .collect import make_player

WIN_TARGET_Z = 1.959964  # two-sided 95%


def _init() -> None:
    torch.set_num_threads(1)


def _play(args) -> dict:
    seed, robot_spec, opp_spec = args
    seat = seed % 4
    players = [make_player(robot_spec if s == seat else opp_spec) for s in range(4)]
    try:
        rec = play_hand(seed, players)
    except IllegalAction:
        return {"seed": seed, "seat": seat, "robot": robot_spec, "illegal": 1, "win": 0, "dealin": 0,
                "exhausted": 0, "winner": "", "win_type": "", "structure": "", "robot_decisions": 0}
    r = rec.result
    return {
        "seed": seed,
        "seat": seat,
        "robot": robot_spec,
        "illegal": rec.illegal,
        "win": int(r.winner == seat),
        "dealin": int(r.win_type == "discard" and r.from_seat == seat),
        "exhausted": int(r.winner is None),
        "winner": "" if r.winner is None else r.winner,
        "win_type": r.win_type or "",
        "structure": r.structure or "",
        "robot_decisions": rec.decisions[seat],
    }


def run_hands(seeds, robot_spec: str, opp_spec: str, processes: int = 20) -> list[dict]:
    with mp.get_context("fork").Pool(processes, initializer=_init) as pool:
        return pool.map(_play, [(s, robot_spec, opp_spec) for s in seeds], chunksize=50)


def paired_diff(a: list[int], b: list[int]) -> dict:
    n = len(a)
    d = [x - y for x, y in zip(a, b)]
    mean = sum(d) / n
    var = sum((x - mean) ** 2 for x in d) / (n - 1)
    half = WIN_TARGET_Z * math.sqrt(var / n)
    return {"mean": mean, "ci95": [mean - half, mean + half], "n": n}


def wilson(k: int, n: int) -> list[float]:
    z = WIN_TARGET_Z
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [c - h, c + h]


def gate1(policy_path: str, out_dir: str, n_hands: int = 10_000, processes: int = 20,
          opponents: str = "heldout_heuristic", test_block: int = 0) -> dict:
    """``test_block`` k uses held-out seeds TEST_BASE + k * 100,000 (block 0 is the original test set)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    seeds = S.test_seeds(n_hands, base=S.TEST_BASE + 100_000 * test_block)
    specs = {"policy": f"policy:{policy_path}", "random": "random", "heuristic": "heuristic"}
    rows = {name: run_hands(seeds, spec, opponents, processes) for name, spec in specs.items()}
    with (out / f"trials_vs_{opponents}.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["condition"] + list(rows["policy"][0].keys()))
        w.writeheader()
        for name, rs in rows.items():
            for r in rs:
                w.writerow({"condition": name, **r})
    summary = {}
    for name, rs in rows.items():
        n = len(rs)
        k = sum(r["win"] for r in rs)
        summary[name] = {
            "hands": n,
            "wins": k,
            "win_rate": k / n,
            "win_rate_ci95": wilson(k, n),
            "dealin_rate": sum(r["dealin"] for r in rs) / n,
            "exhausted_rate": sum(r["exhausted"] for r in rs) / n,
            "illegal_actions": sum(r["illegal"] for r in rs),
            "win_rate_by_seat": [
                sum(r["win"] for r in rs if r["seat"] == s) / max(1, sum(1 for r in rs if r["seat"] == s))
                for s in range(4)
            ],
            "structures": {st: sum(1 for r in rs if r["win"] and r["structure"] == st)
                           for st in ("standard", "seven_pairs", "thirteen_orphans")},
        }
    pw = [r["win"] for r in rows["policy"]]
    vs_random = paired_diff(pw, [r["win"] for r in rows["random"]])
    vs_heur = paired_diff(pw, [r["win"] for r in rows["heuristic"]])
    passed = summary["policy"]["illegal_actions"] == 0 and vs_random["ci95"][0] > 0
    report = {
        "gate": "gate1_playing_agent",
        "policy": policy_path,
        "opponents": opponents,
        "test_seed_range": [seeds[0], seeds[-1]],
        "seat_counts": [sum(1 for s in seeds if s % 4 == k) for k in range(4)],
        "conditions": summary,
        "policy_minus_random": vs_random,
        "policy_minus_heuristic": vs_heur,
        "criteria": {
            "no_illegal_actions": summary["policy"]["illegal_actions"] == 0,
            "advantage_over_random_ci_above_zero": vs_random["ci95"][0] > 0,
        },
        "passed": passed,
    }
    (out / f"report_vs_{opponents}.json").write_text(json.dumps(report, indent=2))
    return report
