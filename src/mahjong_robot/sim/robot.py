"""YAM arm interface in simulation: IK, joint and Cartesian motion, gripper, collision checks.

All targets are for the ``tcp`` site (pad centre) with the gripper body's orientation. The
gripper frame has +z pointing away from the fingertips and y as the closing axis, so the
identity orientation points the fingers straight down with closing along world y.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from .scene import ARM_DAMPING, ARM_JOINTS, ARM_KP, gripper_command_for_gap


def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def down(yaw: float = 0.0, tilt: float = 0.0) -> np.ndarray:
    """Fingers down with the closing axis at ``yaw`` from world y, then ``tilt`` about the closing axis."""
    return rot_z(yaw) @ rot_y(tilt)


def _rot_axis(axis: np.ndarray, a: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


TORQUE_MARGIN = 0.85


class CollisionRisk(RuntimeError):
    pass


@dataclass
class MotionLimits:
    joint_speed: float = 1.2  # rad/s cap for planned joint moves
    cart_speed: float = 0.12  # m/s for Cartesian moves
    fine_speed: float = 0.03  # m/s for contact-phase moves


class Arm:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.m, self.d = model, data
        self.qadr = np.array([model.joint(j).qposadr[0] for j in ARM_JOINTS])
        self.dadr = np.array([model.joint(j).dofadr[0] for j in ARM_JOINTS])
        self.act = np.array([model.actuator(f"{j}_position").id for j in ARM_JOINTS])
        self.grip_act = model.actuator("gripper_position").id
        self.grip_q = model.joint("joint7").qposadr[0]
        self.tcp = model.site("tcp").id
        self.gripper_body = model.body("gripper").id
        lo = np.array([model.joint(j).range[0] for j in ARM_JOINTS])
        hi = np.array([model.joint(j).range[1] for j in ARM_JOINTS])
        self.lo, self.hi = lo + 1e-3, hi - 1e-3
        self._ik_data = mujoco.MjData(model)
        self._rne = np.zeros(model.nv)
        self.torque_limit = np.array([model.actuator(a).forcerange[1] for a in self.act])
        self.limits = MotionLimits()
        arm_bodies = {model.body(n).id for n in ("link1", "link2", "link3", "link4", "link5", "gripper",
                                                   "tip_left", "tip_right")}
        self.arm_geoms = {g for g in range(model.ngeom) if model.geom_bodyid[g] in arm_bodies
                          and model.geom_contype[g]}
        self.finger_geoms = {model.geom(n).id for n in ("tip_left_pad_contact", "tip_right_pad_contact",
                                                         "tip_left_finger_contact", "tip_right_finger_contact")}
        self.pad_geoms = {model.geom(n).id for n in ("tip_left_pad_contact", "tip_right_pad_contact")}
        self.table_geom = model.geom("table").id
        self.free_space = False
        self.label = "motion"
        # False while holding an upright tile: a half turn would reverse the tile's face.
        self.symmetric = True
        # Velocity feedforward cancels the lag that joint damping causes under position control.
        self.ff = np.array(ARM_DAMPING) / np.array(ARM_KP)
        self.step_hook = None  # called after every physics step (e.g. recording)
        self.allowed_contact = set()  # geom ids the non-finger arm may touch in the current move

    # ------------------------------------------------------------ state

    @property
    def q(self) -> np.ndarray:
        return self.d.qpos[self.qadr].copy()

    def tcp_pose(self, data: mujoco.MjData | None = None):
        d = data or self.d
        return d.site_xpos[self.tcp].copy(), d.xmat[self.gripper_body].reshape(3, 3).copy()

    def commanded(self) -> np.ndarray:
        q = getattr(self, "q_cmd", None)
        return (self.d.ctrl[self.act].copy() if q is None else q.copy())

    def sync_command(self) -> None:
        """Adopt the current control targets as the commanded pose (after external resets)."""
        self.q_cmd = self.d.ctrl[self.act].copy()

    @property
    def gap(self) -> float:
        return 2.0 * float(self.d.qpos[self.grip_q])

    # ------------------------------------------------------------ IK

    def ik(self, pos, rot, q0=None, iters: int = 200, tol: float = 2e-4, rot_weight: float = 0.3):
        d = self._ik_data
        d.qpos[:] = self.d.qpos
        q = (self.q if q0 is None else np.array(q0, dtype=float)).copy()
        jac_p = np.zeros((3, self.m.nv))
        jac_r = np.zeros((3, self.m.nv))
        lam = 1e-3
        err_norm = np.inf
        for _ in range(iters):
            d.qpos[self.qadr] = q
            mujoco.mj_kinematics(self.m, d)
            mujoco.mj_comPos(self.m, d)
            p, R = d.site_xpos[self.tcp], d.xmat[self.gripper_body].reshape(3, 3)
            e_p = pos - p
            e_r = 0.5 * (np.cross(R[:, 0], rot[:, 0]) + np.cross(R[:, 1], rot[:, 1]) + np.cross(R[:, 2], rot[:, 2]))
            err = np.concatenate([e_p, rot_weight * e_r])
            err_norm = np.linalg.norm(e_p) + 0.01 * np.linalg.norm(e_r)
            if np.linalg.norm(e_p) < tol and np.linalg.norm(e_r) < 5e-3:
                break
            mujoco.mj_jacSite(self.m, d, jac_p, jac_r, self.tcp)
            J = np.vstack([jac_p[:, self.dadr], rot_weight * jac_r[:, self.dadr]])
            dq = J.T @ np.linalg.solve(J @ J.T + lam * np.eye(6), err)
            q = np.clip(q + np.clip(dq, -0.2, 0.2), self.lo, self.hi)
        ok = np.linalg.norm(e_p) < 1e-3 and np.linalg.norm(e_r) < 2e-2 and self.holdable(q)
        return q, ok, err_norm

    def holdable(self, q) -> bool:
        """Static gravity torque within TORQUE_MARGIN of every joint's actuator limit."""
        d = self._ik_data
        d.qpos[self.qadr] = q
        d.qvel[:] = 0.0
        mujoco.mj_kinematics(self.m, d)
        mujoco.mj_comPos(self.m, d)
        mujoco.mj_rne(self.m, d, 0, self._rne)
        return bool(np.all(np.abs(self._rne[self.dadr]) < TORQUE_MARGIN * self.torque_limit))

    # ------------------------------------------------------------ execution

    def _step(self, n: int = 1) -> None:
        for _ in range(n):
            mujoco.mj_step(self.m, self.d)
            if self.step_hook is not None:
                self.step_hook()
        self._check_collisions()

    def _check_collisions(self) -> None:
        """Stand-in for a joint-torque collision monitor.

        Links other than the fingers may never touch the scene. In free space (``self.free_space``)
        the fingers may touch only a tile clamped between both pads (the held tile).
        """
        d = self.d
        pad_touch: dict[int, set[int]] = {}
        finger_hits = []
        for i in range(d.ncon):
            c = d.contact[i]
            for a, b in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
                if a not in self.arm_geoms or b in self.arm_geoms:
                    continue
                if a not in self.finger_geoms:
                    if b not in self.allowed_contact:
                        raise CollisionRisk(f"{self.m.geom(a).name} touched {self.m.geom(b).name} during {self.label}")
                    continue
                if b == self.table_geom:
                    if c.dist < -0.002 or self.free_space:
                        raise CollisionRisk(f"fingers pressed into the table during {self.label}")
                    continue
                if a in self.pad_geoms:
                    pad_touch.setdefault(b, set()).add(a)
                finger_hits.append(b)
        if self.free_space:
            held = {g for g, pads in pad_touch.items() if len(pads) == 2}
            for b in finger_hits:
                if b not in held:
                    raise CollisionRisk(f"fingers touched {self.m.geom(b).name} in free space during {self.label}")

    def set_gripper_gap(self, gap: float, squeeze: float = 0.0) -> None:
        self.d.ctrl[self.grip_act] = gripper_command_for_gap(gap - squeeze)

    def gripper(self, gap: float, squeeze: float = 0.0, settle: float = 0.35) -> None:
        self.set_gripper_gap(gap, squeeze)
        self._step(int(settle / self.m.opt.timestep))

    def hold(self, seconds: float) -> None:
        self._step(max(1, int(seconds / self.m.opt.timestep)))

    @staticmethod
    def nearest_equivalent(rot, ref):
        """The parallel grasp is symmetric under a half turn about the tool axis; pick the copy nearest ``ref``."""
        flipped = rot @ rot_z(np.pi)
        return rot if np.trace(ref.T @ rot) >= np.trace(ref.T @ flipped) else flipped

    def solve(self, pos, rot, symmetric: bool = True, q_ref=None):
        """Global IK: several seeds, and the 180-degree flip of a symmetric parallel grasp.

        Returns the feasible solution closest to ``q_ref`` (default: the commanded pose), or None.
        """
        q_ref = self.commanded() if q_ref is None else np.asarray(q_ref)
        rots = [rot, rot @ rot_z(np.pi)] if symmetric else [rot]
        yaw0 = np.arctan2(pos[1], pos[0])
        seeds = [q_ref]
        for q2 in (1.0, 1.6, 2.2):
            for q3 in (0.8, 1.4):
                for q4 in (-1.2, -0.5):
                    seeds.append(np.array([yaw0, q2, q3, q4, 0.0, 0.0]))
        best, best_cost = None, np.inf
        w = np.array([1.0, 1.0, 1.0, 0.6, 0.6, 0.3])
        for R in rots:
            for sd in seeds:
                q, ok, _ = self.ik(pos, R, q0=sd, iters=150)
                if ok:
                    cost = float(np.sum(w * np.abs(q - q_ref)))
                    if cost < best_cost:
                        best, best_cost = q, cost
            if best is not None and R is rots[0] and best_cost < 1.0:
                break
        return best

    def move_joints(self, q_target, speed: float | None = None, settle: float = 0.15) -> None:
        q0 = self.commanded()
        q_target = np.clip(q_target, self.lo, self.hi)
        dist = np.max(np.abs(q_target - q0))
        T = max(0.2, dist / (speed or self.limits.joint_speed))
        n = int(T / self.m.opt.timestep)
        dt = self.m.opt.timestep
        for k in range(1, n + 1):
            s = k / n
            pos = 10 * s**3 - 15 * s**4 + 6 * s**5
            vel = (30 * s**2 - 60 * s**3 + 30 * s**4) / (n * dt)
            self._command(q0 + pos * (q_target - q0), vel * (q_target - q0))
            self._step()
        self._command(q_target, 0.0)
        self.hold(settle)

    def _command(self, q, qdot) -> None:
        self.q_cmd = np.asarray(q, dtype=float).copy()
        self.d.ctrl[self.act] = self.q_cmd + self.ff * qdot

    def plan_path(self, poses, q_start=None, max_jump: float = 0.35):
        """Chain local IK through (pos, rot) poses; returns joint waypoints or None if any step fails."""
        q = self.commanded() if q_start is None else np.asarray(q_start, dtype=float)
        out = []
        for pos, rot in poses:
            q2, ok, _ = self.ik(pos, rot, q0=q, iters=200)
            if not ok or np.max(np.abs(q2 - q)) > max_jump:
                return None
            out.append(q2)
            q = q2
        return out

    def follow(self, qs, seconds_per_waypoint: float = 0.12, settle: float = 0.1) -> None:
        """Track precomputed joint waypoints with linear interpolation and velocity feedforward."""
        q_prev = self.commanded()
        steps = max(1, int(seconds_per_waypoint / self.m.opt.timestep))
        for q in qs:
            qdot = (q - q_prev) / (steps * self.m.opt.timestep)
            for j in range(1, steps + 1):
                self._command(q_prev + (j / steps) * (q - q_prev), qdot)
                self._step()
            q_prev = q
        self._command(q_prev, 0.0)
        self.hold(settle)

    def move_to(self, pos, rot, speed: float | None = None, settle: float = 0.4, symmetric: bool | None = None) -> bool:
        """Joint-space move to a TCP pose (free-space motion)."""
        q = self.solve(pos, rot, self.symmetric if symmetric is None else symmetric)
        if q is None:
            return False
        self.move_joints(q, speed, settle)
        return True

    def move_linear(self, pos, rot, speed: float | None = None, settle: float = 0.1, step_len: float = 0.004) -> bool:
        """Straight-line TCP motion with IK at each waypoint."""
        speed = speed or self.limits.cart_speed
        p0, R0 = self.tcp_pose()
        # Interpolate from the commanded (not measured) pose to avoid creeping under load.
        qc = self.commanded()
        dc = self._ik_data
        dc.qpos[self.qadr] = qc
        mujoco.mj_kinematics(self.m, dc)
        p0, R0 = self.tcp_pose(dc)
        if self.symmetric:
            rot = self.nearest_equivalent(rot, R0)
        dist = np.linalg.norm(pos - p0)
        n_way = max(1, int(np.ceil(dist / step_len)))
        dt_way = (dist / n_way) / speed if dist > 0 else 0.05
        q_prev = qc
        for k in range(1, n_way + 1):
            s = k / n_way
            p = p0 + s * (pos - p0)
            R = _slerp_rot(R0, rot, s)
            q, ok, _ = self.ik(p, R, q0=q_prev, iters=60)
            if not ok or np.max(np.abs(q - q_prev)) > 0.3:  # a jump would sweep between branches
                return False
            steps = max(1, int(max(dt_way, 0.004) / self.m.opt.timestep))
            qdot = (q - q_prev) / (steps * self.m.opt.timestep)
            for j in range(1, steps + 1):
                self._command(q_prev + (j / steps) * (q - q_prev), qdot)
                self._step()
            q_prev = q
        self._command(q_prev, 0.0)
        self.hold(settle)
        return True


def _slerp_rot(R0, R1, s):
    dR = R0.T @ R1
    angle = np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1))
    if angle < 1e-6:
        return R1
    axis = np.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1]]) / (2 * np.sin(angle))
    return R0 @ _rot_axis(axis, s * angle)


def look_at_rotation(cam_pos, target, roll: float = 0.0) -> np.ndarray:
    """MuJoCo camera frame (looks along -z, image up = +y) pointing from cam_pos at target."""
    f = np.asarray(target, float) - np.asarray(cam_pos, float)
    f /= np.linalg.norm(f)
    up = np.array([0.0, 0.0, 1.0])
    x = np.cross(f, up)
    if np.linalg.norm(x) < 1e-6:
        x = np.array([1.0, 0.0, 0.0])
    x /= np.linalg.norm(x)
    y = np.cross(x, f)
    R = np.column_stack([x, y, -f])
    return R @ rot_z(roll)
