"""Gate 3: the full decision-to-execution loop in the physics simulator.

Each held-out hand is played with the robot seat driven by perception, the playing agent,
and tile operations; the other seats are played by the table. A hand is *completed* when it
reaches a win or wall exhaustion with no aborting fault and no ground-truth discrepancy.
Ground truth is used here only to score (ADR 0002): after every robot action the evaluator
checks that the physical tiles match what the robot reported.

Modes select the contact controller: ``classical`` (scripted, open-loop from the overhead
estimate), ``hybrid`` (planned free-space motion plus the learned wrist-camera policy near
contact), ``learned`` (the learned policy drives the whole operation).
"""

from __future__ import annotations

import json
import multiprocessing as mp
import time
import traceback
from pathlib import Path

import numpy as np

from ..game import actions as A
from ..game.engine import IllegalAction
from ..game.tiles import type_of
from ..rl import seeds as S
from . import layout as L
from .robot import CollisionRisk
from .robot_system import Fault, RobotSystem
from .table import SimTable
from .world import TableWorld

DEFAULT_CLASSIFIER = "models/tile_classifier.pt"
DEFAULT_CONTACT = "models/contact_policy.pt"
OPPONENT = "heuristic"
MAX_ROBOT_ACTIONS = 60


def make_contact(mode: str, rng):
    from .skills import ScriptedContact

    if mode == "classical":
        return ScriptedContact()
    from .learned import LearnedContact

    return LearnedContact(DEFAULT_CONTACT, full=(mode == "learned"), rng=rng)


def _nearest_active(world: TableWorld, pos: np.ndarray) -> int:
    act = np.flatnonzero(world.active)
    d = [np.linalg.norm(world.tile_pose(i)[0][:2] - pos[:2]) for i in act]
    return int(act[int(np.argmin(d))])


def play_hand(seed: int, mode: str, policy_path: str, classifier_path: str = DEFAULT_CLASSIFIER,
              record_dir: str | None = None, observer=None) -> dict:
    """Play one held-out hand. ``observer(world, table, robot)`` may return an object with
    ``finish(result)``; it watches the hand (e.g. sim/video.py) without affecting it."""
    from ..agents.policy import load_policy
    from ..rl.collect import make_player
    from ..vision.classifier import TileClassifier

    t0 = time.time()
    rng = np.random.default_rng(seed)
    world = TableWorld(seed)
    seat = seed % 4
    table = SimTable(world, seed, seat, [make_player(OPPONENT) for _ in range(3)], rng)
    robot = RobotSystem(world, load_policy(policy_path), TileClassifier(classifier_path), make_contact(mode, rng),
                        rng, seat)
    out = {"seed": seed, "mode": mode, "seat": seat, "completed": False, "fault": None, "detail": "",
           "robot_actions": [], "outcome": None}
    watcher = observer(world, table, robot) if observer is not None else None
    pending_meld = None
    last_claim = None
    try:
        table.deal()
        robot.start()
        for _ in range(MAX_ROBOT_ACTIONS * 3):
            sig = table.advance()
            robot.apply_events(sig.events)
            if sig.kind == "over":
                break
            # A claim is resolved once every claimant has answered, possibly after the robot's
            # commit: the table's announcement is the meld appearing (checked by meld count).
            if last_claim is not None and len(table.engine.melds[seat]) > len(robot.b.own_melds):
                pending_meld, last_claim = last_claim, None
            if pending_meld is not None:
                robot.build_meld(*pending_meld)
                _check_meld(world, robot, table)
                table.meld_built()
                pending_meld = None
            if sig.kind == "turn":
                a = robot.on_turn()
            elif sig.kind == "after_call":
                a = robot.on_after_call()
            else:
                a = robot.on_claim(sig.discarder)
                claimed_type = robot.b.current_discard[1]
            if a not in table.legal():
                raise Fault("belief", f"robot chose {A.describe(a)}, not legal in the engine "
                                      f"(engine hand {table.robot_hand_types()}, belief {sorted(robot.b.counts_list())})")
            new = table.commit(a)
            out["robot_actions"].append(A.describe(a))
            if a == A.WIN_DISCARD:
                robot.won_by_discard()
            if a < A.CONCEALED_KONG:
                _check_discard(world, robot, table, a - A.DISCARD, new)
            if sig.kind == "claim":
                last_claim = (a, claimed_type) if a in (A.PUNG, A.KONG, A.CHOW_LOW, A.CHOW_MID, A.CHOW_HIGH) else None
        else:
            raise Fault("planning", "hand exceeded the action budget")
        res = table.engine.result
        out["outcome"] = ("robot_win" if res.winner == seat else "exhausted" if res.winner is None else "other_win")
        out["win_type"] = res.win_type if res.winner == seat else None
        belief = sorted(robot.b.counts_list())
        truth = table.robot_hand_types()
        if belief != truth:
            raise Fault("silent_identity", f"final belief {belief} != engine hand {truth}")
        out["completed"] = True
    except Fault as f:
        out["fault"], out["detail"] = f.category, f.detail
    except IllegalAction as e:
        out["fault"], out["detail"] = "belief", str(e)
    except CollisionRisk as e:  # raised by free-space motion outside a tile operation (e.g. parking)
        out["fault"], out["detail"] = "collision", str(e)
    except Exception as e:  # noqa: BLE001 - every failure must be recorded, not crash the gate
        out["fault"], out["detail"] = "software", f"{type(e).__name__}: {e} | {traceback.format_exc(limit=3)}"
    if out["fault"] and record_dir:
        _dump(Path(record_dir) / f"{mode}_{seed}", world, robot, out)
    out["sim_time_s"] = round(float(world.d.time), 1)
    out["wall_time_s"] = round(time.time() - t0, 1)
    out.update({k: v for k, v in robot.stats.items()})
    if watcher is not None:
        watcher.finish(out)
    robot.close()
    return out


def _dump(path: Path, world, robot, out: dict) -> None:
    """Failure evidence: last overhead frame, a fresh overhead frame, and ground-truth tile states."""
    from PIL import Image

    path.mkdir(parents=True, exist_ok=True)
    fr = getattr(robot, "last_frame", None)
    if fr is not None:
        Image.fromarray(fr.rgb).save(path / "last_overhead.png")
    try:
        Image.fromarray(robot.overhead.capture(world.d).rgb).save(path / "at_fault_overhead.png")
    except Exception:  # noqa: BLE001
        pass
    gt = []
    for i in np.flatnonzero(world.active):
        p, _ = world.tile_pose(i)
        gt.append({"inst": int(i), "type": int(type_of(i)), "pos": np.round(p, 4).tolist(),
                   "upright": world.is_upright(i), "face_up": world.is_face_up(i)})
    dets = [{"pos": np.round(d.pos, 4).tolist(), "upright": d.upright, "type": d.tile_type,
             "conf": round(d.confidence, 2), "region": d.region} for d in getattr(robot, "last_dets", [])]
    belief = {"hand": [(round(h.x, 4), h.tile_type) for h in robot.b.hand], "drawn": robot.b.drawn}
    (path / "state.json").write_text(json.dumps({"result": out, "ground_truth": gt, "detections": dets,
                                                 "belief": belief}, indent=1, default=str))


def _check_discard(world, robot, table, t, new_events):
    """Ground truth: the tile in the robot's newest pond slot has the reported type."""
    slot = L.pond_pos(robot.b.pond_slot - 1)
    phys = _nearest_active(world, slot)
    truth = type_of(phys)
    if truth != t:
        raise Fault("silent_identity", f"robot discarded a {truth} while reporting {t}")
    for e in new_events:
        if e[0] == "discard" and e[1] == table.seat:
            table.record_robot_discard(e[2], phys)


def _check_meld(world, robot, table):
    m = len(robot.b.own_melds) - 1
    meld = robot.b.own_melds[m]
    for k, inst in enumerate(meld.tiles):
        phys = _nearest_active(world, L.meld_pos(m, k))
        if type_of(phys) != type_of(inst):
            raise Fault("silent_identity", f"meld slot {k} holds a {type_of(phys)}, belief {type_of(inst)}")


def _worker(args):
    import torch

    torch.set_num_threads(1)
    return play_hand(*args)


def run_gate3(policy: str, n_hands: int, mode: str, out_dir: str, processes: int = 20, start: int = 0,
              seed_base: int = S.SIM_TEST_BASE, record: bool = True) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    seeds = S.test_seeds(n_hands, base=seed_base + start)
    rec = str(out / "faults") if record else None
    with mp.get_context("spawn").Pool(processes) as pool:
        rows = pool.map(_worker, [(s, mode, policy, DEFAULT_CLASSIFIER, rec) for s in seeds], chunksize=1)
    with (out / f"trials_{mode}.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r, default=lambda o: o.item() if hasattr(o, "item") else str(o)) + "\n")
    summary = summarize(rows)
    rep = {"gate": "gate3_physics_integration", "mode": mode, "policy": policy, "seeds": [seeds[0], seeds[-1]],
           "summary": summary, "passed": summary["completed"] >= 0.7 * n_hands}
    (out / f"report_{mode}.json").write_text(json.dumps(rep, indent=2))
    return rep


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    done = [r for r in rows if r["completed"]]
    faults: dict[str, int] = {}
    for r in rows:
        if r["fault"]:
            faults[r["fault"]] = faults.get(r["fault"], 0) + 1
    return {
        "hands": n,
        "completed": len(done),
        "completion_rate": len(done) / max(1, n),
        "faults": faults,
        "outcomes": {k: sum(1 for r in done if r["outcome"] == k) for k in ("robot_win", "other_win", "exhausted")},
        "mean_sim_time_s_completed": float(np.mean([r["sim_time_s"] for r in done])) if done else None,
        "mean_operations": float(np.mean([r["operations"] for r in rows])),
        "recoveries": int(sum(r["recoveries"] for r in rows)),
        "hands_with_recovery_completed": sum(1 for r in done if r["recoveries"] > 0),
    }
