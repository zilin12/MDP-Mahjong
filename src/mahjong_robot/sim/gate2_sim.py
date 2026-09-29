"""Simulation counterpart of gate 2's per-skill trials (the gate itself is physical).

For each tile operation, run held-out randomized trials whose targets come from the overhead
camera (never ground truth), with a chosen contact controller, and score them with ground
truth. Seeds are disjoint from the contact policy's training seeds.
"""

from __future__ import annotations

import json
import multiprocessing as mp
from pathlib import Path

import numpy as np

from . import layout as L
from .cameras import D435I, SimCamera
from .perception import detect_tiles
from .robot import CollisionRisk
from .robot_system import GAP_TOL
from .skills import Operations, TileEstimate
from .world import TableWorld

SKILLS = ("shared_zone_pick", "close_packed_extraction", "insertion", "reorientation", "placement")
SEED_BASE = 50_000  # development trials
FINAL_SEED_BASE = 60_000  # final held-out trials (never used during development)


def _json_default(o):
    return o.item() if hasattr(o, "item") else str(o)


def _estimate(world, cam, ops, near_xy, upright, radius=0.02):
    ops.park()
    dets = detect_tiles(cam.capture(world.d))
    if hasattr(ops.contact, "overhead_bias"):
        for d in dets:
            d.pos[:2] += ops.contact.overhead_bias()
    cands = [d for d in dets if d.upright == upright and np.linalg.norm(d.pos[:2] - near_xy) < radius]
    if len(cands) != 1:
        return None
    d = cands[0]
    return TileEstimate(d.pos, d.yaw, upright)


def _row(world, rng, n, right_end: float = L.HAND_ANCHOR_X):
    ids = [int(i) for i in rng.permutation(136)[:n + 2]]
    xs = []
    for k in range(n):
        x = right_end - (n - 1 - k) * (L.TILE_W + 0.0003) + rng.normal(0, 0.0003)
        world.activate(ids[k], np.array([x, L.HAND_Y, L.TILE_H / 2 + 0.0003]) + [0, rng.normal(0, 0.0006), 0],
                       L.yaw_quat(rng.normal(0, 0.01), L.UPRIGHT_QUAT))
        xs.append(x)
    return ids, xs


def _present(world, rng, inst, upright):
    xy = L.SHARED_XY + rng.uniform(-L.SHARED_JITTER_XY, L.SHARED_JITTER_XY, 2)
    yaw = rng.uniform(-L.SHARED_JITTER_YAW, L.SHARED_JITTER_YAW)
    if upright:
        world.activate(inst, np.array([xy[0], xy[1], L.TILE_H / 2 + 0.0003]), L.yaw_quat(yaw, L.UPRIGHT_QUAT))
    else:
        world.activate(inst, np.array([xy[0], xy[1], L.TILE_T / 2 + 0.0003]), L.yaw_quat(yaw))


def _snapshot(world, exclude):
    return {int(i): (world.tile_pose(i)[0].copy(), world.is_upright(i)) for i in np.flatnonzero(world.active)
            if int(i) not in exclude}


def _disturbed(world, before, tol=GAP_TOL):
    bad = []
    for i, (p, up) in before.items():
        q = world.tile_pose(i)[0]
        moved = float(np.linalg.norm(q[:2] - p[:2]))
        toppled = up and not world.is_upright(i)
        if moved > tol or toppled:
            bad.append((i, round(moved * 1000, 1), bool(toppled)))
    return bad


def trial(skill: str, seed: int, mode: str) -> dict:
    from .gate3 import make_contact

    rng = np.random.default_rng(seed)
    world = TableWorld(seed)
    ops = Operations(world.arm, make_contact(mode, rng))
    cam = SimCamera(world.m, D435I, rng, 0.001, 0.1)
    out = {"skill": skill, "seed": seed, "mode": mode, "success": False, "reason": ""}
    try:
        # Insertion trials alternate ends; for the right end the row sits left, as when the robot
        # chooses that end because it has more room.
        right_end = L.HAND_ANCHOR_X if (skill != "insertion" or seed % 2 == 0) else L.HAND_ANCHOR_X - 0.04
        ids, xs = _row(world, rng, 13 if skill != "insertion" else 12, right_end)
        extra = ids[-1]
        if skill == "shared_zone_pick":
            upright = bool(rng.random() < 0.5)
            _present(world, rng, extra, upright)
            world.settle(0.4)
            est = _estimate(world, cam, ops, L.SHARED_XY, upright)
            if est is None:
                out["reason"] = "not perceived"
                return out
            before = _snapshot(world, {extra})
            ops.pick(est)
            held = world.in_gripper(extra) and world.tile_pose(extra)[0][2] > 0.04
            bad = _disturbed(world, before)
            out.update(success=held and not bad, reason="" if held else "not held", disturbed=len(bad))
            out["variant"] = "upright" if upright else "face_up"
        elif skill == "close_packed_extraction":
            world.settle(0.4)
            k = int(rng.integers(0, 13))
            est = _estimate(world, cam, ops, np.array([xs[k], L.HAND_Y]), True, radius=0.008)
            if est is None:
                out["reason"] = "not perceived"
                return out
            before = _snapshot(world, {ids[k]})
            ops.pick(est)
            held = world.in_gripper(ids[k]) and world.tile_pose(ids[k])[0][2] > 0.04
            bad = _disturbed(world, before)
            out.update(success=held and not bad, reason="" if held and not bad else ("not held" if not held else "neighbours disturbed"),
                       disturbed=bad)
        elif skill == "insertion":
            _present(world, rng, extra, True)
            world.settle(0.4)
            est = _estimate(world, cam, ops, L.SHARED_XY, True)
            if est is None:
                out["reason"] = "not perceived"
                return out
            ops.pick(est)
            side = "left" if seed % 2 == 0 else "right"  # alternate ends, as in play
            end_k = 0 if side == "left" else len(xs) - 1
            row_end = _estimate(world, cam, ops, np.array([xs[end_k], L.HAND_Y]), True, radius=0.008)
            if row_end is None:
                out["reason"] = "not perceived"
                return out
            before = _snapshot(world, {extra})
            sign = -1.0 if side == "left" else 1.0
            target = np.array([row_end.pos[0] + sign * (L.TILE_W + L.INSERT_MARGIN), L.HAND_Y, 0.0])
            ops.insert(target, 0.0, side)
            world.settle(0.3)
            p = world.tile_pose(extra)[0]
            left_gap = sign * (p[0] - world.tile_pose(ids[end_k])[0][0]) - L.TILE_W
            out["variant"] = side
            ok = world.is_upright(extra) and -0.0005 < left_gap < 0.006 and abs(p[1] - L.HAND_Y) < 0.004
            bad = _disturbed(world, before)
            out.update(success=ok and not bad, reason="" if ok and not bad else ("misplaced" if not ok else "neighbours disturbed"),
                       gap_mm=round(left_gap * 1000, 2))
        elif skill in ("reorientation", "placement"):
            flat = skill == "placement" and rng.random() < 0.5
            _present(world, rng, extra, not flat)
            world.settle(0.4)
            est = _estimate(world, cam, ops, L.SHARED_XY, not flat)
            if est is None:
                out["reason"] = "not perceived"
                return out
            ops.pick(est)
            if flat:
                slot = L.meld_pos(int(rng.integers(0, 4)), 0)
                ops.place_flat(slot, 0.0)
                out["variant"] = "meld_flat"
            else:
                slot = L.pond_pos(int(rng.integers(0, 24))) if rng.random() < 0.7 else L.meld_pos(int(rng.integers(0, 4)), int(rng.integers(1, 4)))
                ops.locate_staged = lambda: _estimate(world, cam, ops, L.STAGING_XY, True)
                ops.tip_over(slot)
                out["variant"] = "tip_over"
            world.settle(0.3)
            err = float(np.linalg.norm(world.tile_pose(extra)[0][:2] - slot[:2]))
            face_up = world.is_face_up(extra)
            ok = face_up if skill == "reorientation" else (face_up and err < 0.006)
            out.update(success=ok, reason="" if ok else ("not face-up" if not face_up else "off slot"),
                       slot_err_mm=round(err * 1000, 2))
    except CollisionRisk as e:
        out["reason"] = f"collision risk: {e}"
    finally:
        cam.close()
    return out


def _worker(args):
    import torch

    torch.set_num_threads(1)
    return trial(*args)


def run(out_dir: str, modes=("classical", "hybrid"), n: int = 50, processes: int = 20, seed_base: int = SEED_BASE) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(skill, seed_base + 1000 * si + t, mode) for mode in modes for si, skill in enumerate(SKILLS) for t in range(n)]
    with mp.get_context("spawn").Pool(processes) as pool:
        rows = pool.map(_worker, jobs, chunksize=1)
    with (out / "skill_trials_sim.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r, default=_json_default) + "\n")
    summary = {}
    for mode in modes:
        summary[mode] = {}
        for skill in SKILLS:
            rs = [r for r in rows if r["mode"] == mode and r["skill"] == skill]
            k = int(sum(bool(r["success"]) for r in rs))
            reasons: dict[str, int] = {}
            for r in rs:
                if not r["success"]:
                    reasons[r["reason"][:60]] = reasons.get(r["reason"][:60], 0) + 1
            summary[mode][skill] = {"successes": k, "trials": len(rs), "rate": k / max(1, len(rs)),
                                    "passes_90pct": k >= 0.9 * len(rs), "failures": reasons}
    (out / "skill_trials_sim_summary.json").write_text(json.dumps(summary, indent=2))
    return summary
