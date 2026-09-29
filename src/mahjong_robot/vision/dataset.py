"""Render labelled tile-face crops for the classifier (simulation only).

Two views, matching run-time use: face-up tiles seen by the overhead D435i, and upright
tiles seen face-on by a D405-like camera from the robot side. Labels come from simulator
ground truth, which is legitimate for training data (ADR 0002 forbids it only at run time).
Crops are cut from perturbed poses, like real detections, so the classifier tolerates
misaligned crops.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from ..game.tiles import N_TYPES
from ..sim import layout as L
from ..sim.cameras import D405, D435I, CameraSpec, SimCamera
from ..sim.perception import Detection, crop_flat, crop_upright_face
from ..sim.robot import look_at_rotation
from ..sim.world import TableWorld


def _randomize_lighting(m: mujoco.MjModel, rng: np.random.Generator) -> None:
    m.vis.headlight.ambient[:] = rng.uniform(0.2, 0.5)
    m.vis.headlight.diffuse[:] = rng.uniform(0.3, 0.6)
    m.light_diffuse[0] = rng.uniform(0.35, 0.75, 3)
    m.light_dir[0] = np.array([rng.normal(0, 0.3), rng.normal(0, 0.3), -1.0])


def _types_for(rng, n):
    return rng.integers(0, N_TYPES, n)


def render_flat_views(world: TableWorld, rng: np.random.Generator, n_scenes: int, per_scene: int = 30):
    """Face-up tiles scattered over the table, cropped from overhead frames."""
    X, y = [], []
    m, d = world.m, world.d
    world.arm.move_joints(np.array([3.0, 1.9, 1.9, -1.6, 0.0, 0.0]), settle=0.05)  # out of view
    cam = SimCamera(m, D435I, rng, calib_trans_sigma=0.001, calib_rot_sigma_deg=0.1)
    for _ in range(n_scenes):
        _randomize_lighting(m, rng)
        cam.recalibrate(0.001, 0.1)
        for i in np.flatnonzero(world.active):
            world.deactivate(i)
        types = _types_for(rng, per_scene)
        cells = rng.choice(14 * 17, per_scene, replace=False)
        used = []
        for k, (t, c) in enumerate(zip(types, cells)):
            inst = int(t) * 4 + (k % 4)
            if world.active[inst]:
                continue
            gx, gy = c % 14, c // 14  # covers the whole visible table, x in [-0.28, 0.27], y in [0.10, 0.90]
            pos = np.array([-0.28 + gx * 0.042, 0.10 + gy * 0.05, L.TILE_T / 2 + 0.0003])
            pos[:2] += rng.normal(0, 0.004, 2)
            yaw = rng.uniform(-np.pi, np.pi)
            world.activate(inst, pos, L.yaw_quat(yaw))
            used.append((inst, int(t)))
        mujoco.mj_forward(m, d)
        frame = cam.capture(d)
        for inst, t in used:
            p, _ = world.tile_pose(inst)
            R = world.tile_rot(inst)
            yaw = np.arctan2(R[1, 0], R[0, 0]) + rng.normal(0, 0.06) + (np.pi if rng.random() < 0.5 else 0.0)
            det = Detection(p + np.r_[rng.normal(0, 0.0015, 2), 0], yaw, False, (0, 0))
            c = crop_flat(frame, det)
            if c is not None:
                X.append(c)
                y.append(t)
    cam.close()
    return X, y


def render_face_views(world: TableWorld, rng: np.random.Generator, n_scenes: int, per_scene: int = 14):
    """Upright rows seen face-on from the robot side with a D405-like camera."""
    X, y = [], []
    m, d = world.m, world.d
    mount = m.body("data_cam_mount").mocapid[0]
    spec = CameraSpec("data_cam", D405.width, D405.height, D405.baseline)
    world.arm.move_joints(np.array([3.0, 1.9, 1.9, -1.6, 0.0, 0.0]), settle=0.05)
    cam = SimCamera(m, spec, rng, calib_trans_sigma=0.0005, calib_rot_sigma_deg=0.3)
    for _ in range(n_scenes):
        _randomize_lighting(m, rng)
        cam.recalibrate(0.0005, 0.3)
        for i in np.flatnonzero(world.active):
            world.deactivate(i)
        types = _types_for(rng, per_scene)
        row_y = rng.uniform(0.15, 0.25)
        x0 = rng.uniform(-0.2, -0.1)
        used = []
        for k, t in enumerate(types):
            inst = int(t) * 4 + (k % 4)
            if world.active[inst]:
                continue
            pos = np.array([x0 + (k + 0.5) * L.TILE_W + rng.normal(0, 0.0003), row_y, L.TILE_H / 2 + 0.0003])
            world.activate(inst, pos, L.yaw_quat(rng.normal(0, 0.03), L.UPRIGHT_QUAT))
            used.append((inst, int(t)))
        centre = np.array([x0 + per_scene * L.TILE_W / 2, row_y, 0.013])
        cam_pos = centre + np.array([rng.uniform(-0.15, 0.15), -rng.uniform(0.09, 0.20), rng.uniform(0.04, 0.13)])
        Rcam = look_at_rotation(cam_pos, centre + np.r_[rng.normal(0, 0.06), 0, 0], roll=rng.choice([0.0, np.pi]))
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, Rcam.flatten())
        d.mocap_pos[mount] = cam_pos
        d.mocap_quat[mount] = q
        mujoco.mj_forward(m, d)
        frame = cam.capture(d)
        for inst, t in used:
            p, _ = world.tile_pose(inst)
            R = world.tile_rot(inst)
            yaw = np.arctan2(R[1, 0], R[0, 0]) + rng.normal(0, 0.03)
            c = crop_upright_face(frame, p + np.r_[rng.normal(0, 0.001, 2), 0], yaw)
            if c is not None:
                X.append(c)
                y.append(t)
    cam.close()
    return X, y


def _part(args):
    seed, flat_scenes, face_scenes = args
    import torch

    torch.set_num_threads(1)
    rng = np.random.default_rng(seed)
    world = TableWorld(seed)
    Xf, yf = render_flat_views(world, rng, flat_scenes)
    Xu, yu = render_face_views(world, rng, face_scenes)
    return Xf, yf, Xu, yu


def build_parallel(out: Path, seed: int, flat_scenes: int, face_scenes: int, procs: int = 12) -> Path:
    import multiprocessing as mp

    jobs = [(seed * 1000 + i, flat_scenes // procs, face_scenes // procs) for i in range(procs)]
    with mp.get_context("spawn").Pool(procs) as pool:
        parts = pool.map(_part, jobs)
    Xf = [x for p in parts for x in p[0]]
    yf = [t for p in parts for t in p[1]]
    Xu = [x for p in parts for x in p[2]]
    yu = [t for p in parts for t in p[3]]
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, X=np.stack(Xf + Xu), y=np.array(yf + yu), view=np.array([0] * len(yf) + [1] * len(yu)))
    return out


def build(out: Path, seed: int, flat_scenes: int, face_scenes: int) -> Path:
    rng = np.random.default_rng(seed)
    world = TableWorld(seed)
    Xf, yf = render_flat_views(world, rng, flat_scenes)
    Xu, yu = render_face_views(world, rng, face_scenes)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, X=np.stack(Xf + Xu), y=np.array(yf + yu),
                        view=np.array([0] * len(yf) + [1] * len(yu)))
    return out
