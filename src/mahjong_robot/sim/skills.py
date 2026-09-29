"""Tile operations (ADR 0004) split into free-space motion and a contact phase.

Every operation takes perceived estimates, never simulator ground truth. Free-space motion
goes to a pre-contact pose with a joint-space move. The contact phase is delegated to a
``ContactController``: ``ScriptedContact`` runs the geometric motion open-loop from the
estimate (classical), and the learned controller (sim/learned.py) corrects it from the
wrist camera.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import layout as L
from .robot import Arm, CollisionRisk, down, rot_x, rot_z

SAFE_Z = 0.075  # TCP height for transfers above the table
HIGH_Z = 0.16  # height for joint-space fallback moves
PRE_Z = 0.045  # pre-contact hover height above the contact pose
GRIP_SQUEEZE = 0.004
UPRIGHT_GRASP_Z = L.TILE_H / 2
FLAT_GRASP_Z = 0.012
PUSH_Z = 0.013
TIP_ROLL = np.deg2rad(60.0)  # release a few degrees short of flat
TIP_PRETILTS = tuple(np.deg2rad(a) for a in (45.0, 35.0, 55.0, 25.0))
TIP_GRASP_OFFSET = np.array([0.0, 0.002, 0.0045])  # tile frame: toward the top, toward the face
TIP_WAYPOINTS = 24


@dataclass
class TileEstimate:
    pos: np.ndarray  # tile centre (m)
    yaw: float  # rotation about z of the tile's width axis from world x
    upright: bool
    tile_type: int | None = None
    confidence: float = 1.0


@dataclass
class OpResult:
    ok: bool
    reason: str = ""
    sim_time: float = 0.0
    corrections: list = field(default_factory=list)


def upright_grasp_rot(yaw: float) -> np.ndarray:
    return down(yaw)  # closing along the tile's thickness


def flat_grasp_rot(yaw: float) -> np.ndarray:
    return down(yaw + np.pi / 2)  # closing along the tile's width


class ContactController:
    """Strategy for the contact phase. ``target`` is the TCP contact pose from the estimate."""

    name = "base"

    def descend(self, ops: "Operations", target: np.ndarray, rot: np.ndarray, skill: str) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


class ScriptedContact(ContactController):
    name = "classical"

    def descend(self, ops, target, rot, skill):
        ops.lin(target, rot, speed=ops.arm.limits.fine_speed, what=f"{skill} contact")
        return target, rot


class Operations:
    def __init__(self, arm: Arm, contact: ContactController | None = None):
        self.arm = arm
        self.contact = contact or ScriptedContact()
        self.held_upright: bool | None = None
        self.face_in_gripper: np.ndarray | None = None
        self.locate_staged = None  # optional callback returning a TileEstimate of the staged tile

    def lin(self, pos, rot, speed=0.05, settle=0.05, what="move") -> None:
        """Straight-line move that must succeed; falls back to a joint move if local IK fails."""
        if self.arm.move_linear(np.asarray(pos, dtype=float), rot, speed=speed, settle=settle):
            return
        if not self.arm.move_to(np.asarray(pos, dtype=float), rot, speed=0.4, settle=settle):
            raise CollisionRisk(f"no IK solution to {what} at {np.round(pos, 3)}")

    # ------------------------------------------------------------ free space

    def transfer(self, pos: np.ndarray, rot: np.ndarray) -> None:
        """Free-space motion: lift, travel at SAFE_Z while re-orienting, descend to ``pos``.

        Straight Cartesian segments keep the fingers above the tiles; a joint-space move is used
        only as a fallback, and only at HIGH_Z. Any finger contact except with the held tile
        is a collision (Arm.free_space).
        """
        arm = self.arm
        if getattr(self.contact, "full", False):
            return self.contact.transfer(self, np.asarray(pos, dtype=float), rot)  # learned-only ablation
        arm.free_space = True
        try:
            p, R = arm.tcp_pose()
            arm.free_space = False  # the first lift may still brush the tile just released
            if p[2] < SAFE_Z - 0.005:
                arm.label = "transfer lift"
                lift = np.array([p[0], p[1], SAFE_Z])
                if not arm.move_linear(lift, R, speed=0.08, settle=0.05):
                    # Tilted wrist (after a tip-over): back out along the tool axis, then lift high.
                    arm.move_linear(p + R[:, 2] * 0.02, R, speed=0.04, settle=0.05)
                    if not arm.move_to(np.array([p[0], p[1], HIGH_Z]), rot, settle=0.05):
                        raise CollisionRisk(f"cannot lift to transfer height at {np.round(p, 3)}")
            arm.free_space = True
            arm.label = "transfer travel"
            p, R = arm.tcp_pose()
            travel_z = max(SAFE_Z, p[2]) if p[2] < HIGH_Z - 0.005 else HIGH_Z
            above = np.array([pos[0], pos[1], max(travel_z, pos[2])])
            if not arm.move_linear(above, rot, speed=0.15, settle=0.1, step_len=0.01):
                arm.label = "transfer fallback"
                high = np.array([p[0], p[1], HIGH_Z])
                arm.move_linear(high, arm.tcp_pose()[1], speed=0.1, settle=0.05)
                if not arm.move_to(np.array([pos[0], pos[1], HIGH_Z]), rot, settle=0.1):
                    raise CollisionRisk(f"no IK solution above {np.round(pos, 3)}")
            p, _ = arm.tcp_pose()
            if pos[2] < p[2] - 1e-4:
                arm.label = "transfer descend"
                self.lin(pos, rot, speed=0.08, settle=0.35, what="descend to hover")
        finally:
            arm.free_space = False

    def park(self) -> None:
        """Move out of the overhead camera's view of the work areas."""
        self.transfer(L.PARK_XYZ, down(0.0))

    # ------------------------------------------------------------ operations

    def pick(self, est: TileEstimate) -> None:
        """Shared-zone pick or close-packed extraction, depending on the estimate."""
        arm = self.arm
        if est.upright:
            rot, z, width = upright_grasp_rot(est.yaw), UPRIGHT_GRASP_Z, L.TILE_T
            skill = "extract"
        else:
            rot, z, width = flat_grasp_rot(est.yaw), FLAT_GRASP_Z, L.TILE_W
            skill = "pick_flat"
        target = np.array([est.pos[0], est.pos[1], z])
        arm.set_gripper_gap(width + 0.012)
        # Upright tiles are always taken with the canonical flip (face along the gripper's -y):
        # it is the one whose joint-6 range covers the hand, the shared zone and the staging spot.
        arm.symmetric = not est.upright
        self.transfer(target + [0, 0, PRE_Z], rot)
        target, rot = self.contact.descend(self, target, rot, skill)
        arm.gripper(width, squeeze=GRIP_SQUEEZE, settle=0.4)
        if est.upright:
            # Hand and presented tiles face the robot (ADR 0004); remember the face in the gripper frame.
            face_world = rot_z(est.yaw) @ np.array([0.0, -1.0, 0.0])
            self.face_in_gripper = arm.tcp_pose()[1].T @ face_world
        self.lin(target + [0, 0, PRE_Z], rot, speed=0.04, what="lift the tile")
        self.held_upright = est.upright

    def face_preserving(self, rot: np.ndarray, yaw: float) -> np.ndarray:
        """Of the two equivalent grasps, the one that keeps the held tile's face toward the robot."""
        want = rot_z(yaw) @ np.array([0.0, -1.0, 0.0])
        flipped = rot @ rot_z(np.pi)
        f = self.face_in_gripper
        return rot if (rot @ f) @ want >= (flipped @ f) @ want else flipped

    def _released(self) -> None:
        self.held_upright = None
        self.face_in_gripper = None
        self.arm.symmetric = True

    def insert(self, slot: np.ndarray, yaw: float = 0.0, side: str = "left") -> None:
        """Lower the held upright tile against the ``side`` end of the row, then release."""
        arm = self.arm
        rot = self.face_preserving(upright_grasp_rot(yaw), yaw)
        target = np.array([slot[0], slot[1], UPRIGHT_GRASP_Z + 0.0015])
        self.transfer(target + [0, 0, PRE_Z], rot)
        target, rot = self.contact.descend(self, target, rot, f"insert_{side}")
        arm.gripper(L.TILE_T + 0.010, settle=0.3)
        self._released()
        self.lin(target + [0, 0, PRE_Z], rot, speed=0.05, settle=0.05)

    def tip_over(self, slot: np.ndarray) -> None:
        """Reorient the held upright tile face-up so it lands centred on ``slot`` (ADR 0004).

        Stand the tile at the staging spot and regrasp it by its now-free sides with the
        wrist tilted toward the robot; carry it upright to the stand position in front of
        the slot; roll it backward about its bottom-back edge along a precomputed joint
        path; release a few degrees short of flat.
        """
        arm = self.arm
        stand = np.array([slot[0], slot[1] - (L.TILE_T + L.TILE_H) / 2, UPRIGHT_GRASP_Z])
        plan = self._plan_lay(stand)
        if plan is None:
            raise CollisionRisk(f"no feasible lay-down path at {np.round(slot, 3)}")
        side, grasp_at_stand, path = plan
        # 1. Stand the tile at the staging spot (face toward the robot).
        rot = self.face_preserving(upright_grasp_rot(0.0), 0.0)
        stage = np.array([L.STAGING_XY[0], L.STAGING_XY[1], UPRIGHT_GRASP_Z + 0.0015])
        self.transfer(stage + [0, 0, PRE_Z], rot)
        stage, rot = self.contact.descend(self, stage, rot, "stand")
        arm.gripper(L.TILE_T + 0.012, settle=0.3)
        self._released()
        self.lin(stage + [0, 0, PRE_Z], rot, speed=0.05, settle=0.05)
        # 2. Regrasp by the sides with the same tilt and flip the roll will use, at the staged
        #    tile's perceived position when a locator is available.
        offset = grasp_at_stand - stand
        staged = np.array([stage[0], stage[1], UPRIGHT_GRASP_Z])
        if self.locate_staged is not None:
            est = self.locate_staged()
            if est is not None:
                staged = np.array([est.pos[0], est.pos[1], UPRIGHT_GRASP_Z])
        g = staged + offset
        arm.set_gripper_gap(L.TILE_W + 0.008)
        self.transfer(g + [0, 0, PRE_Z], side)
        g, side = self.contact.descend(self, g, side, "regrasp")
        arm.gripper(L.TILE_W, squeeze=GRIP_SQUEEZE, settle=0.4)
        # 3. Carry it to the stand position and set it down (1 mm above the table).
        self.lin(g + [0, 0, PRE_Z], side, speed=0.04, what="lift for carry")
        arm.symmetric = False  # keep the planned flip
        try:
            self.transfer(grasp_at_stand + [0, 0, PRE_Z], side)
            dest, side = self.contact.descend(self, grasp_at_stand + [0, 0, 0.001], side, "set_down")
            arm.hold(0.2)
            # 4. Roll along the planned path, release, back out.
            if not np.allclose(arm.commanded(), path[0], atol=0.3):
                path = self._plan_lay(stand, fixed=side)[2]
            roll, back = path[:TIP_WAYPOINTS], path[TIP_WAYPOINTS:]
            arm.follow(roll, seconds_per_waypoint=0.12, settle=0.1)
            arm.gripper(L.TILE_W + 0.008, settle=0.4)
            arm.follow(back, seconds_per_waypoint=0.1, settle=0.05)
        finally:
            self._released()

    def _plan_lay(self, stand: np.ndarray, fixed: np.ndarray | None = None):
        """Find a pre-tilt and grasp flip whose whole roll is reachable without jumps.

        Returns (side-grasp rotation, TCP at the stand, joint path of roll + back-out).
        """
        arm = self.arm
        Rt = rot_x(np.pi / 2)  # upright tile frame
        g = np.array(stand) + Rt @ TIP_GRASP_OFFSET
        pivot = np.array([stand[0], stand[1] + L.TILE_T / 2, 0.0])
        r0 = g - pivot
        candidates = [fixed] if fixed is not None else [
            rot_x(pre) @ down(np.pi / 2) @ rot_z(flip) for pre in TIP_PRETILTS for flip in (0.0, np.pi)]
        for side in candidates:
            q0 = arm.solve(g + [0, 0, 0.001], side, symmetric=False)
            if q0 is None:
                continue
            poses = []
            for s in np.linspace(0, 1, TIP_WAYPOINTS + 1)[1:]:
                R = rot_x(-TIP_ROLL * s)
                poses.append((pivot + R @ r0 + [0, 0, 0.001 + 0.001 * s], R @ side))
            end_p, end_R = poses[-1]
            for k in range(1, 7):  # back out along the tool axis
                poses.append((end_p + end_R[:, 2] * 0.004 * k + [0, 0, 0.003 * k], end_R))
            top = poses[-1][0]
            n_up = max(1, int(np.ceil((SAFE_Z - top[2]) / 0.008)))
            for k in range(1, n_up + 1):  # then straight up to transfer height
                poses.append((top + [0, 0, (SAFE_Z - top[2]) * k / n_up], end_R))
            path = arm.plan_path(poses, q_start=q0)
            if path is not None:
                self._last_plan = np.round(side[:, 2], 2).tolist()
                return side, g, path
        return None

    def place_flat(self, slot: np.ndarray, yaw: float = 0.0) -> None:
        arm = self.arm
        rot = flat_grasp_rot(yaw)
        target = np.array([slot[0], slot[1], FLAT_GRASP_Z + 0.0015])
        self.transfer(target + [0, 0, PRE_Z], rot)
        target, rot = self.contact.descend(self, target, rot, "place_flat")
        arm.gripper(L.TILE_W + 0.012, settle=0.3)
        self._released()
        self.lin(target + [0, 0, PRE_Z], rot, speed=0.05, settle=0.05)

    def push(self, x_start: float, x_end: float, y: float) -> None:
        """Compaction: sweep the closed fingers along the row from ``x_start`` to ``x_end``."""
        arm = self.arm
        arm.set_gripper_gap(0.0)
        rot = down(0.0)
        back = -0.004 if x_end > x_start else 0.004
        start = np.array([x_start, y, PUSH_Z])
        self.transfer(start + [0, 0, PRE_Z], rot)
        self.lin(start, rot, speed=0.03, settle=0.1, what="push start")
        self.lin(np.array([x_end, y, PUSH_Z]), rot, speed=0.01, settle=0.3, what="push")
        self.lin(np.array([x_end + back, y, PUSH_Z]), rot, speed=0.02, settle=0.05, what="push back-off")
        self.lin(np.array([x_end + back, y, PUSH_Z + PRE_Z]), rot, speed=0.05, settle=0.05, what="push lift")
