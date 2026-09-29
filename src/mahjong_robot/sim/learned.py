"""Learned visuomotor contact policy (the "learned" half of the hybrid system).

``ContactNet`` maps a wrist D405 RGB-D crop around the tool point, plus the skill, to the
correction (dx, dy, dyaw) that brings the TCP onto the true contact pose, in the gripper's
horizontal frame. ``LearnedContact`` runs it closed-loop at the pre-contact hover and again
halfway down, instead of descending open-loop on the overhead estimate.

With ``full=True`` (the learned-only ablation), ``ApproachNet`` also replaces the planned
free-space motion. It is a goal-conditioned policy imitating the planner's demonstrations
(lift, travel, descend) that outputs bounded Cartesian steps, and the contact net then
drives the descent.

Training data is rendered in simulation with ground-truth labels (legitimate for training;
ADR 0002 forbids ground truth only at run time). The hover offsets are sampled to cover the
overhead estimation error.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import mujoco
import numpy as np
import torch
from torch import nn

from . import layout as L
from .cameras import D405, SimCamera
from .robot import CollisionRisk, down, rot_z
from .skills import PRE_Z, SAFE_Z, ContactController, flat_grasp_rot, upright_grasp_rot

SKILLS = ("extract", "pick_flat", "insert_left", "insert_right", "regrasp")
# The hybrid system refines a skill with the learned policy only if, on held-out validation
# samples, 90% of its residuals are below this (it must beat the open-loop overhead estimate).
HYBRID_P90_MAX_MM = 3.0
CROP = 96
CROP_SRC = 200  # pixels of the 640 x 360 wrist frame around the tool point
MAX_STEP = 0.004
ITERS = 3


def wrist_crop(frame, point_world: np.ndarray) -> np.ndarray:
    """RGB-D crop centred on the projection of ``point_world``: 4 x CROP x CROP float32."""
    u, v = frame.project(point_world)
    h, w = frame.depth.shape
    half = CROP_SRC // 2
    u0, v0 = int(round(u)) - half, int(round(v)) - half
    pad = CROP_SRC
    rgb = cv2.copyMakeBorder(frame.rgb, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    dep = cv2.copyMakeBorder(frame.depth, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    rgb = rgb[v0 + pad : v0 + pad + CROP_SRC, u0 + pad : u0 + pad + CROP_SRC]
    dep = dep[v0 + pad : v0 + pad + CROP_SRC, u0 + pad : u0 + pad + CROP_SRC]
    rgb = cv2.resize(rgb, (CROP, CROP), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    dep = cv2.resize(dep, (CROP, CROP), interpolation=cv2.INTER_NEAREST)
    # Depth relative to the tool point's own depth, in centimetres.
    pc = (np.asarray(point_world) - frame.t_wc) @ frame.R_wc
    rel = np.where(dep > 0, (dep - pc[2]) * 100.0, 0.0).astype(np.float32)
    return np.concatenate([rgb.transpose(2, 0, 1), rel[None]], 0)


class ContactNet(nn.Module):
    def __init__(self):
        super().__init__()

        def blk(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, stride=2, padding=1), nn.BatchNorm2d(o), nn.ReLU(),
                                 nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU())

        self.cnn = nn.Sequential(blk(4, 32), blk(32, 64), blk(64, 128), blk(128, 128), nn.AdaptiveAvgPool2d(1),
                                 nn.Flatten())
        self.skill = nn.Embedding(len(SKILLS), 16)
        self.head = nn.Sequential(nn.Linear(128 + 16 + 1, 128), nn.ReLU(), nn.Linear(128, 3))

    def forward(self, img, skill, height):
        z = torch.cat([self.cnn(img), self.skill(skill), height[:, None]], 1)
        return self.head(z)  # (dx mm, dy mm, dyaw deg)


class ApproachNet(nn.Module):
    """Goal-conditioned free-space policy: (goal - tcp, rotation error, tcp z, phase) -> step."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(17, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(),
                                 nn.Linear(256, 6))

    def forward(self, x):
        return self.net(x)


def grasp_frame(R: np.ndarray, skill: str) -> tuple[np.ndarray, float]:
    """Horizontal frame of a grasp and the tile yaw it is aligned with.

    The closing axis is horizontal in every grasp (tilts are about it), so its heading
    defines a 2-D frame. Upright grasps close across the thickness (tile width axis is
    perpendicular to the closing axis); flat grasps and the side regrasp close across the width.
    """
    phi = float(np.arctan2(R[1, 1], R[0, 1]))
    F = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
    tile_yaw = phi - np.pi / 2 if skill.startswith(("extract", "insert")) else phi
    return F, tile_yaw


def _rot_err(R, Rg):
    return 0.5 * (np.cross(R[:, 0], Rg[:, 0]) + np.cross(R[:, 1], Rg[:, 1]) + np.cross(R[:, 2], Rg[:, 2]))


def approach_features(p, R, goal_p, goal_R, lifted: float, q, symmetric: bool = True) -> np.ndarray:
    """Goal-conditioned state for ApproachNet.

    Includes the rotation error to both equivalent goal orientations (when the grasp is
    symmetric) and the joint positions, so the policy can learn which equivalent the planner
    reaches within the joint limits.
    """
    err_a = _rot_err(R, goal_R)
    err_b = _rot_err(R, goal_R @ rot_z(np.pi)) if symmetric else err_a
    return np.concatenate([(goal_p - p) * 10.0, err_a, err_b, [p[2] * 10.0, lifted], np.asarray(q) / np.pi]).astype(np.float32)


APPROACH_DIM = 3 + 3 + 3 + 2 + 6


class LearnedContact(ContactController):
    name = "learned"

    def __init__(self, path: str | Path, full: bool = False, rng=None, device: str = "cpu"):
        ck = torch.load(path, map_location=device, weights_only=False)
        self.net = ContactNet()
        self.net.load_state_dict(ck["contact"])
        self.net.eval()
        self.full = full
        self.val_p90 = ck.get("val_p90_mm", {})
        # Learned-only uses the policy for every contact phase; the hybrid only where validated.
        # The hybrid refines only the skills its development trials validated (stored with the
        # model); without that record, fall back to the held-out residual rule.
        validated = ck.get("hybrid_refined")
        if full:
            self.refined = set(SKILLS)
        elif validated is not None:
            self.refined = set(validated)
        else:
            self.refined = {k for k, v in self.val_p90.items() if v < HYBRID_P90_MAX_MM}
        self.approach = None
        if full:
            self.approach = ApproachNet()
            self.approach.load_state_dict(ck["approach"])
            self.approach.eval()
        self.cam = None
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.corrections: list[float] = []
        self.deltas: list[np.ndarray] = []  # refined target minus overhead-estimated target

    def _camera(self, ops):
        if self.cam is None:
            self.cam = SimCamera(ops.arm.m, D405, self.rng, 0.0005, 0.3)
        return self.cam

    @torch.no_grad()
    def _predict(self, ops, target, skill):
        fr = self._camera(ops).capture(ops.arm.d)
        img = torch.from_numpy(wrist_crop(fr, target))[None]
        p, _ = ops.arm.tcp_pose()
        out = self.net(img, torch.tensor([SKILLS.index(skill)]), torch.tensor([float((p[2] - target[2]) * 100)]))
        dx, dy, dyaw = out[0].numpy()
        return np.array([dx, dy]) / 1000.0, np.deg2rad(dyaw)

    def descend(self, ops, target, rot, skill):
        arm = ops.arm
        if skill not in self.refined:
            ops.lin(target, rot, speed=arm.limits.fine_speed, what=f"{skill} contact")
            return target, rot
        target = np.array(target, dtype=float)
        start = target.copy()
        # A symmetric grasp may be executed as its half-turned equivalent: work in the actual frame.
        rot = arm.nearest_equivalent(rot, arm.tcp_pose()[1]) if arm.symmetric else rot
        # Second refinement stage: as low as possible while the pads (11 mm below the TCP) stay
        # above neighbouring tiles' tops, so lateral corrections never push a neighbour.
        low = 0.017 if skill == "pick_flat" else 0.027
        for stage_h in (PRE_Z, low):
            for _ in range(ITERS):
                d_xy, d_yaw = self._predict(ops, target, skill)
                F, _ = grasp_frame(arm.tcp_pose()[1], skill)
                step = F @ d_xy
                n = np.linalg.norm(step)
                if n > MAX_STEP:
                    step *= MAX_STEP / n
                d_yaw = float(np.clip(d_yaw, -0.08, 0.08))
                target[:2] += step
                rot = rot_z(d_yaw) @ rot
                self.corrections.append(float(n))
                ops.lin(target + [0, 0, stage_h], rot, speed=0.02, settle=0.25, what=f"{skill} refine")
                if n < 0.0004 and abs(d_yaw) < 0.005:
                    break
        self.deltas.append(target[:2] - start[:2])
        ops.lin(target, rot, speed=arm.limits.fine_speed, what=f"{skill} contact")
        return target, rot

    def overhead_bias(self) -> np.ndarray:
        """Median wrist-camera correction of overhead-estimated targets: the overhead extrinsic's
        local error, fed back into later overhead estimates (online calibration)."""
        if len(self.deltas) < 1:
            return np.zeros(2)
        return np.median(np.array(self.deltas[-15:]), axis=0)

    def transfer(self, ops, pos, rot):
        """Learned-only free-space motion: roll out ApproachNet toward ``pos`` (no planner)."""
        arm = ops.arm
        arm.free_space = True
        lifted = 0.0
        try:
            goal_R = (_reachable_goal_rot(arm, pos, rot, arm.commanded()) if arm.symmetric else rot)
            if goal_R is None:
                raise CollisionRisk("learned approach: goal orientation unreachable")
            for _ in range(240):
                p, R = arm.tcp_pose()
                if np.linalg.norm(pos - p) < 0.003 and np.linalg.norm(_rot_err(R, goal_R)) < 0.05:
                    return
                x = torch.from_numpy(approach_features(p, R, pos, goal_R, lifted, arm.q, symmetric=False))[None]
                with torch.no_grad():
                    a = self.approach(x)[0].numpy()
                if p[2] > SAFE_Z - 0.01:
                    lifted = 1.0
                for scale in (1.0, 0.5, 0.25):
                    dp = np.clip(a[:3] / 100.0, -0.02, 0.02) * scale
                    dr = np.clip(a[3:], -0.15, 0.15) * scale
                    ang = np.linalg.norm(dr)
                    Rn = R if ang < 1e-6 else _axis_angle(dr / ang, ang) @ R
                    if arm.move_linear(p + dp, Rn, speed=0.15, settle=0.0, step_len=0.01):
                        break
                else:
                    raise CollisionRisk("learned approach left the reachable workspace")
            raise CollisionRisk("learned approach did not converge")
        finally:
            arm.free_space = False


def _axis_angle(k, a):
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


# ---------------------------------------------------------------- data generation


def _sample_offset(rng):
    """Hover offsets covering the overhead estimate's error, including its occasional large misses."""
    if rng.random() < 0.3:
        return rng.uniform(-0.008, 0.008, 2), np.deg2rad(rng.uniform(-10, 10))
    return rng.normal(0, 0.0028, 2), np.deg2rad(rng.normal(0, 4))


def _scene(world, rng, skill):
    """Lay out a hand row plus the skill's target; returns (target body or None, row order)."""
    n = int(rng.integers(8, 15))
    order = [int(i) for i in rng.permutation(136)[: n + 1]]
    x_right = rng.uniform(L.HAND_ANCHOR_X - 0.04, L.HAND_ANCHOR_X)
    for k in range(n):
        x = x_right - (n - 1 - k) * (L.TILE_W + 0.0003) + rng.normal(0, 0.0004)
        world.activate(order[k], np.array([x, L.HAND_Y + rng.normal(0, 0.0008), L.TILE_H / 2 + 0.0003]),
                       L.yaw_quat(rng.normal(0, 0.01), L.UPRIGHT_QUAT))
    extra = order[n]
    target = order[int(rng.integers(0, n))]
    if skill == "pick_flat":
        xy = L.SHARED_XY + rng.uniform(-0.006, 0.006, 2)
        world.activate(extra, np.array([xy[0], xy[1], L.TILE_T / 2 + 0.0003]), L.yaw_quat(rng.uniform(-0.14, 0.14)))
        target = extra
    elif skill == "regrasp":
        xy = L.STAGING_XY + rng.normal(0, 0.0015, 2)
        world.activate(extra, np.array([xy[0], xy[1], L.TILE_H / 2 + 0.0003]),
                       L.yaw_quat(rng.normal(0, 0.03), L.UPRIGHT_QUAT))
        target = extra
    elif skill == "extract" and rng.random() < 0.3:  # a drawn tile standing in the shared zone
        xy = L.SHARED_XY + rng.uniform(-0.006, 0.006, 2)
        world.activate(extra, np.array([xy[0], xy[1], L.TILE_H / 2 + 0.0003]),
                       L.yaw_quat(rng.uniform(-0.14, 0.14), L.UPRIGHT_QUAT))
        target = extra
    world.settle(0.4)
    return target, order, n


def _true_pose(world, skill, target):
    """Ground-truth contact TCP position, tile yaw, and nominal grasp rotation."""
    from .robot import rot_x
    from .skills import TIP_GRASP_OFFSET, TIP_PRETILTS

    tp, _ = world.tile_pose(target)
    R = world.tile_rot(target)
    yaw = float(np.arctan2(R[1, 0], R[0, 0]))
    if skill == "pick_flat":
        return np.array([tp[0], tp[1], 0.012]), yaw, flat_grasp_rot(yaw)
    if skill == "regrasp":
        g = tp + rot_z(yaw) @ rot_x(np.pi / 2) @ TIP_GRASP_OFFSET
        # The pre-tilt and flip vary per scene (keyed by the staged tile's pose) so the policy
        # sees every grasp the tip-over planner may choose.
        key = abs(int(round(tp[0] * 1e5))) * 7 + abs(int(round(tp[1] * 1e5))) + target
        pre = TIP_PRETILTS[key % len(TIP_PRETILTS)]
        flip = np.pi * ((key // 7) % 2)
        return g, yaw, rot_z(yaw) @ rot_x(pre) @ down(np.pi / 2) @ rot_z(flip)
    return np.array([tp[0], tp[1], L.TILE_H / 2]), yaw, upright_grasp_rot(yaw)


def generate_contact_data(seed: int, n_scenes: int, per_scene: int = 24):
    """Render (crop, skill, height, label) samples at perturbed pre-contact poses."""
    from .skills import Operations, TileEstimate
    from .world import TableWorld

    rng = np.random.default_rng(seed)
    X, S, H, Y = [], [], [], []
    world = TableWorld(seed)
    arm = world.arm
    ops = Operations(arm)
    cam = SimCamera(world.m, D405, rng, 0.0005, 0.3)
    for scene in range(n_scenes):
        for i in np.flatnonzero(world.active):
            world.deactivate(i)
        world.reset_arm()
        skill = SKILLS[scene % len(SKILLS)]
        try:
            target, order, n = _scene(world, rng, skill)
            if skill.startswith("insert"):
                # Hold a presented tile over a free row end: the target is one tile width plus
                # the insertion margin beyond the end tile (ADR 0004).
                held = order[n]
                world.activate(held, np.array([*L.SHARED_XY, L.TILE_H / 2 + 0.0003]), L.UPRIGHT_QUAT)
                world.settle(0.3)
                ops.pick(TileEstimate(world.tile_pose(held)[0], 0.0, True))
                if skill == "insert_left":
                    end, sign = world.tile_pose(order[0])[0], -1.0
                else:
                    end, sign = world.tile_pose(order[n - 1])[0], 1.0
                true = np.array([end[0] + sign * (L.TILE_W + L.INSERT_MARGIN), end[1], L.TILE_H / 2 + 0.0015])
                true_yaw, base_rot = 0.0, ops.face_preserving(upright_grasp_rot(0.0), 0.0)
            for _ in range(per_scene):
                if not skill.startswith("insert"):
                    true, true_yaw, base_rot = _true_pose(world, skill, target)
                off, off_yaw = _sample_offset(rng)
                cmd = true + np.array([off[0], off[1], rng.uniform(0.012, PRE_Z)])
                rot = rot_z(off_yaw) @ base_rot
                # Same grasp-flip policy as at run time: upright tiles (extract, insert) always use
                # the canonical flip; flat picks and the side regrasp may use either equivalent.
                canonical = skill.startswith(("extract", "insert"))
                if not arm.move_to(cmd, rot, settle=0.3, symmetric=not canonical):
                    continue
                fr = cam.capture(world.d)
                p, Rg = arm.tcp_pose()
                F, g_yaw = grasp_frame(Rg, skill)
                err_xy = F.T @ (true[:2] - p[:2])
                d_yaw = (true_yaw - g_yaw + np.pi / 2) % np.pi - np.pi / 2
                X.append(wrist_crop(fr, np.array([p[0], p[1], true[2]])))
                S.append(SKILLS.index(skill))
                H.append((p[2] - true[2]) * 100)
                Y.append([err_xy[0] * 1000, err_xy[1] * 1000, np.degrees(d_yaw)])
        except CollisionRisk:
            world.reset_arm()
        ops._released()
    cam.close()
    return np.stack(X).astype(np.float16), np.array(S), np.array(H, np.float32), np.array(Y, np.float32)


def generate_approach_data(seed: int, n_paths: int):
    """Imitation data for ApproachNet from the planner's transfer (lift, travel, descend).

    Each path starts from a random pose (reached with an unrecorded planner move) or from the
    end of the previous path, and goes to a random work-area goal. Samples pair the state
    with the planner's displacement over the next 50 ms.
    """
    from .skills import Operations
    from .world import TableWorld

    rng = np.random.default_rng(seed)
    world = TableWorld(seed)
    ops = Operations(world.arm)
    arm = world.arm
    Xs, Ys = [], []
    goals = ([np.r_[L.HAND_ANCHOR_X - k * L.TILE_W, L.HAND_Y, 0.013] for k in range(15)]
             + [L.pond_pos(k) for k in range(24)] + [np.r_[xy, 0.013] for xy in L.MELD_SLOTS]
             + [np.r_[L.SHARED_XY, 0.013], np.r_[L.STAGING_XY, 0.013], L.PARK_XYZ])
    for _ in range(n_paths):
        if rng.random() < 0.5:
            start = np.array([rng.uniform(-0.25, 0.25), rng.uniform(0.12, 0.35), rng.uniform(0.04, 0.2)])
            try:
                arm.move_to(start, down(rng.uniform(-np.pi, np.pi)), settle=0.1)
            except CollisionRisk:
                world.reset_arm()
        g = goals[int(rng.integers(len(goals)))].copy()
        if g[2] < 0.1:
            g[2] = rng.uniform(0.013, 0.06)
        R_goal = down(rng.uniform(-0.2, 0.2) + (np.pi / 2 if rng.random() < 0.3 else 0.0))
        traj = []
        k = [0]

        def hook():
            k[0] += 1
            if k[0] % 25 == 0:
                traj.append((*arm.tcp_pose(), arm.q))

        arm.step_hook = hook
        try:
            ops.transfer(g, R_goal)
        except CollisionRisk:
            world.reset_arm()
        arm.step_hook = None
        lifted = 0.0
        for a, b in zip(traj[:-1], traj[1:]):
            (p, R, q), (p2, R2, _) = a, b
            if p[2] > SAFE_Z - 0.01:
                lifted = 1.0
            Xs.append(approach_features(p, R, g, R_goal, lifted, q))
            Ys.append(np.concatenate([(p2 - p) * 100.0, _rot_err(R, R2)]))
    return np.array(Xs, np.float32), np.array(Ys, np.float32)


def train_approach(data: Path, checkpoint: Path, epochs: int = 80, device: str | None = None) -> float:
    """Train ApproachNet alone and store it in an existing checkpoint."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    z = np.load(data)
    AX, AY = torch.from_numpy(z["AX"]).to(device), torch.from_numpy(z["AY"]).to(device)
    ap = ApproachNet().to(device)
    opt = torch.optim.Adam(ap.parameters(), lr=1e-3)
    for _ in range(epochs):
        perm = torch.randperm(len(AX), device=device)
        for i in range(0, len(perm), 512):
            b = perm[i : i + 512]
            loss = nn.functional.mse_loss(ap(AX[b]), AY[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    ck["approach"] = {k: v.cpu() for k, v in ap.state_dict().items()}
    torch.save(ck, checkpoint)
    return float(loss.detach())


def train_models(data: Path, out: Path, epochs: int = 40, device: str | None = None) -> dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    z = np.load(data)
    X = torch.from_numpy(z["X"].astype(np.float32)).to(device)
    S = torch.from_numpy(z["S"]).long().to(device)
    H = torch.from_numpy(z["H"]).to(device)
    Y = torch.from_numpy(z["Y"]).to(device)
    n = len(X)
    idx = torch.randperm(n, generator=torch.Generator().manual_seed(0)).to(device)
    va, tr = idx[: n // 10], idx[n // 10 :]
    net = ContactNet().to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    hist = []
    for ep in range(epochs):
        net.train()
        perm = tr[torch.randperm(len(tr), device=device)]
        for i in range(0, len(perm), 128):
            b = perm[i : i + 128]
            img = X[b]
            img = torch.cat([img[:, :3] * (1 + 0.2 * (torch.rand(len(b), 1, 1, 1, device=device) - 0.5)), img[:, 3:]], 1)
            loss = nn.functional.smooth_l1_loss(net(img, S[b], H[b]), Y[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        net.eval()
        with torch.no_grad():
            pv = torch.cat([net(X[va[i:i + 256]], S[va[i:i + 256]], H[va[i:i + 256]]) for i in range(0, len(va), 256)])
            err = (pv - Y[va]).abs().mean(0).cpu().numpy()
        hist.append({"epoch": ep, "val_mae_mm_x": float(err[0]), "val_mae_mm_y": float(err[1]), "val_mae_deg": float(err[2])})
    ck = {"contact": {k: v.cpu() for k, v in net.state_dict().items()}}
    with torch.no_grad():
        pv = torch.cat([net(X[va[i:i + 256]], S[va[i:i + 256]], H[va[i:i + 256]]) for i in range(0, len(va), 256)])
        res = torch.linalg.norm(pv[:, :2] - Y[va, :2], dim=1).cpu().numpy()
    sv = S[va].cpu().numpy()
    ck["val_p90_mm"] = {sk: float(np.percentile(res[sv == k], 90)) for k, sk in enumerate(SKILLS) if (sv == k).any()}
    hist.append({"val_p90_mm": ck["val_p90_mm"]})
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ck, out)
    return {"history": hist, "n_samples": n}


# ---------------------------------------------------------------- DAgger for the learned-only approach


def _slerp_step(R, R_goal, max_angle):
    dR = R.T @ R_goal
    angle = np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1))
    if angle < 1e-6:
        return np.zeros(3)
    axis = np.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1]]) / (2 * np.sin(angle))
    return R @ axis * min(angle, max_angle)  # world-frame rotation vector


def expert_step(p, R, goal_p, goal_R):
    """Analytic lift-travel-descend expert: Cartesian step (m) and world rotation vector (rad)."""
    horiz = np.linalg.norm(goal_p[:2] - p[:2])
    travel_z = max(SAFE_Z, goal_p[2])
    if horiz > 0.004 and p[2] < travel_z - 0.004:
        dp = np.array([0.0, 0.0, min(0.012, travel_z - p[2])])
        frac = 0.1
    elif horiz > 0.004:
        d = goal_p[:2] - p[:2]
        step = min(0.015, horiz)
        dp = np.r_[d / horiz * step, np.clip(travel_z - p[2], -0.01, 0.01)]
        frac = step / max(horiz, 1e-6)
    else:
        dz = goal_p[2] - p[2]
        dp = np.r_[goal_p[:2] - p[:2], np.clip(dz, -0.01, 0.01)]
        frac = 1.0
    dR = R.T @ goal_R
    remaining = np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1))
    dr = _slerp_step(R, goal_R, min(0.12, max(remaining * frac, min(remaining, 0.03))))
    return dp, dr


def _reachable_goal_rot(arm, goal_p, goal_R, q_ref):
    """The equivalent goal orientation the planner would reach: IK-feasible and nearest in joints."""
    best, cost = None, np.inf
    for R in (goal_R, goal_R @ rot_z(np.pi)):
        q = arm.solve(goal_p, R, symmetric=False, q_ref=q_ref)
        if q is not None:
            c = float(np.sum(np.abs(q - q_ref)))
            if c < cost:
                best, cost = R, c
    return best


def dagger_approach(seed: int, iterations: int = 3, episodes: int = 600, steps: int = 60, device: str = "cpu"):
    """Train ApproachNet by DAgger with kinematic rollouts (IK only, no physics)."""
    from .world import TableWorld

    rng = np.random.default_rng(seed)
    world = TableWorld(seed)
    arm = world.arm
    goals = ([np.r_[L.HAND_ANCHOR_X - k * L.TILE_W, L.HAND_Y, 0.058] for k in range(15)]
             + [np.r_[L.pond_pos(k)[:2], 0.053] for k in range(24)] + [np.r_[xy, 0.053] for xy in L.MELD_SLOTS]
             + [np.r_[L.SHARED_XY, 0.058], np.r_[L.STAGING_XY, 0.058], L.PARK_XYZ])
    net = ApproachNet()
    X, Y = [], []
    for it in range(iterations):
        beta = 1.0 - it / max(1, iterations - 1)
        for _ in range(episodes):
            start = np.array([rng.uniform(-0.28, 0.2), rng.uniform(0.02, 0.33), rng.uniform(0.04, 0.22)])
            q = arm.solve(start, down(rng.uniform(-np.pi, np.pi)), q_ref=world.HOME_Q)
            if q is None:
                continue
            g = goals[int(rng.integers(len(goals)))].copy()
            g[2] += rng.uniform(-0.01, 0.01)
            g_R = _reachable_goal_rot(arm, g, down(rng.uniform(-0.2, 0.2) + (np.pi / 2 if rng.random() < 0.3 else 0.0)), q)
            if g_R is None:
                continue
            lifted = 0.0
            for _ in range(steps):
                d = arm._ik_data
                d.qpos[arm.qadr] = q
                mujoco.mj_kinematics(arm.m, d)
                p, R = arm.tcp_pose(d)
                if p[2] > SAFE_Z - 0.01:
                    lifted = 1.0
                if np.linalg.norm(g - p) < 0.003 and np.linalg.norm(_rot_err(R, g_R)) < 0.05:
                    break
                dp_e, dr_e = expert_step(p, R, g, g_R)
                x = approach_features(p, R, g, g_R, lifted, q, symmetric=False)
                X.append(x)
                Y.append(np.r_[dp_e * 100.0, dr_e])
                if rng.random() < beta:
                    dp, dr = dp_e, dr_e
                else:
                    with torch.no_grad():
                        a = net(torch.from_numpy(x)[None])[0].numpy()
                    dp, dr = np.clip(a[:3] / 100.0, -0.02, 0.02), np.clip(a[3:], -0.15, 0.15)
                ang = np.linalg.norm(dr)
                Rn = R if ang < 1e-9 else _axis_angle(dr / ang, ang) @ R
                q2, ok, _ = arm.ik(p + dp, Rn, q0=q, iters=80)
                if not ok or np.max(np.abs(q2 - q)) > 0.3:
                    break
                q = q2
        AX, AY = torch.from_numpy(np.array(X, np.float32)), torch.from_numpy(np.array(Y, np.float32))
        net = ApproachNet()
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        for _ in range(60):
            perm = torch.randperm(len(AX))
            for i in range(0, len(perm), 512):
                b = perm[i : i + 512]
                loss = nn.functional.mse_loss(net(AX[b]), AY[b])
                opt.zero_grad()
                loss.backward()
                opt.step()
        net.eval()
    return net, float(loss.detach()), len(X)
