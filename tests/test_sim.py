"""Simulation checks. Physics tests are marked slow (run with `pytest -m slow`)."""

from __future__ import annotations

import random

import numpy as np
import pytest

from mahjong_robot.game.engine import CLAIM, Engine, claim_options, turn_actions
from mahjong_robot.game.tiles import type_of


def test_pure_legality_matches_engine():
    """The robot computes legality from its belief with the engine's own pure functions."""
    rng = random.Random(0)
    for seed in range(100):
        e = Engine(seed)
        while not e.done:
            s = e.to_act
            if e.phase == CLAIM:
                d = e.window.discard
                ref = claim_options(e.counts[s], e.n_melds(s), type_of(d.tile), s == (d.seat + 1) % 4, e.wall_remaining)
            else:
                ref = turn_actions(e.counts[s], e.melds[s], e.wall_remaining, e.phase)
            assert tuple(ref) == e.legal_actions()
            e.step(rng.choice(e.legal_actions()))


@pytest.mark.slow
def test_extraction_and_tip_over_with_true_poses():
    from mahjong_robot.sim import layout as L
    from mahjong_robot.sim.skills import Operations, TileEstimate
    from mahjong_robot.sim.world import TableWorld

    w = TableWorld(0)
    ops = Operations(w.arm)
    for k in range(13):
        w.activate(k, np.array([L.HAND_ANCHOR_X - (12 - k) * 0.0203, L.HAND_Y, L.TILE_H / 2 + 0.0003]), L.UPRIGHT_QUAT)
    w.settle(0.4)
    before = {k: w.tile_pose(k)[0] for k in range(13)}
    p, _ = w.tile_pose(6)
    ops.pick(TileEstimate(p, 0.0, True))
    assert w.in_gripper(6)
    assert all(np.linalg.norm(w.tile_pose(k)[0] - before[k]) < 0.002 for k in range(13) if k != 6)
    slot = L.pond_pos(3)
    ops.tip_over(slot)
    w.settle(0.3)
    assert w.is_face_up(6)
    assert np.linalg.norm(w.tile_pose(6)[0][:2] - slot[:2]) < 0.006


@pytest.mark.slow
def test_overhead_row_detection_is_sub_millimetre():
    from mahjong_robot.sim import layout as L
    from mahjong_robot.sim.cameras import D435I, SimCamera
    from mahjong_robot.sim.perception import detect_tiles
    from mahjong_robot.sim.skills import Operations
    from mahjong_robot.sim.world import TableWorld

    w = TableWorld(1)
    ops = Operations(w.arm)
    rng = np.random.default_rng(1)
    xs, x = [], -0.185
    for k in range(13):
        x += L.TILE_W + (rng.uniform(0, 0.003) if rng.random() < 0.3 else 0.0002)
        xs.append(x)
        w.activate(k * 4, np.array([x, L.HAND_Y, L.TILE_H / 2 + 0.0003]), L.UPRIGHT_QUAT)
    w.settle(0.4)
    ops.park()
    cam = SimCamera(w.m, D435I, rng, 0.0, 0.0)
    dets = sorted((d for d in detect_tiles(cam.capture(w.d)) if d.upright and abs(d.pos[1] - L.HAND_Y) < 0.012),
                  key=lambda d: d.pos[0])
    cam.close()
    gt = sorted(w.tile_pose(k * 4)[0][0] for k in range(13))
    assert len(dets) == 13
    assert np.max(np.abs(np.array([d.pos[0] for d in dets]) - gt)) < 0.001
