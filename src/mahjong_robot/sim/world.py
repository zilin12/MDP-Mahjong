"""Simulated table: tile activation, table-side (simulated human) placement, and ground truth.

Tiles not on the table float in storage with contacts off and gravity compensated. The
"table" (the simulated human of ADR 0003) teleports tiles for dealing, opponents' discards,
and presentations in the shared zone. Ground-truth accessors here are for scoring and for
generating demonstrations only; the robot's runtime path must not call them (ADR 0002).
"""

from __future__ import annotations

import mujoco
import numpy as np

from ..game.tiles import N_INSTANCES
from . import layout as L
from .robot import Arm
from .scene import load_model


class TableWorld:
    def __init__(self, seed: int = 0, model: mujoco.MjModel | None = None):
        self.m = model if model is not None else load_model()
        self.d = mujoco.MjData(self.m)
        self.rng = np.random.default_rng(seed)
        self.body = np.array([self.m.body(f"tile_{i}").id for i in range(N_INSTANCES)])
        self.geom = np.array([self.m.geom(f"tile_{i}_geom").id for i in range(N_INSTANCES)])
        self.qadr = np.array([self.m.joint(f"tile_{i}_free").qposadr[0] for i in range(N_INSTANCES)])
        self.vadr = np.array([self.m.joint(f"tile_{i}_free").dofadr[0] for i in range(N_INSTANCES)])
        self.active = np.zeros(N_INSTANCES, dtype=bool)
        self.arm = Arm(self.m, self.d)
        self.geom_to_tile = {int(g): i for i, g in enumerate(self.geom)}
        self.reset_arm()

    # ------------------------------------------------------------ arm

    HOME_Q = np.array([1.57, 1.2, 1.0, -1.0, 0.0, 0.0])

    def reset_arm(self) -> None:
        a = self.arm
        self.d.qpos[a.qadr] = self.HOME_Q
        self.d.ctrl[a.act] = self.HOME_Q
        a.sync_command()
        self.d.ctrl[a.grip_act] = 0.0
        mujoco.mj_forward(self.m, self.d)

    # ------------------------------------------------------------ tiles

    def activate(self, i: int, pos, quat) -> None:
        """Place tile instance ``i`` on the table (the simulated human's hand)."""
        m, d = self.m, self.d
        b, g = self.body[i], self.geom[i]
        m.body_gravcomp[b] = 0.0
        m.geom_contype[g] = m.geom_conaffinity[g] = 1
        m.body_contype[b] = m.body_conaffinity[b] = 1
        a, v = self.qadr[i], self.vadr[i]
        d.qpos[a : a + 3] = pos
        d.qpos[a + 3 : a + 7] = np.asarray(quat) / np.linalg.norm(quat)
        d.qvel[v : v + 6] = 0.0
        self.active[i] = True
        self._wake(b)

    def deactivate(self, i: int) -> None:
        m, d = self.m, self.d
        b, g = self.body[i], self.geom[i]
        m.body_gravcomp[b] = 1.0
        m.geom_contype[g] = m.geom_conaffinity[g] = 0
        m.body_contype[b] = m.body_conaffinity[b] = 0
        a, v = self.qadr[i], self.vadr[i]
        d.qpos[a : a + 3] = L.storage_pos(i)
        d.qpos[a + 3 : a + 7] = L.FLAT_QUAT
        d.qvel[v : v + 6] = 0.0
        self.active[i] = False

    def _wake(self, body: int) -> None:
        # Teleporting changes qpos directly; a forward pass lets the sleep logic see the new state.
        mujoco.mj_forward(self.m, self.d)

    def settle(self, seconds: float = 0.3) -> None:
        self.arm.hold(seconds)

    # ------------------------------------------------------------ ground truth (scoring only)

    def tile_pose(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        a = self.qadr[i]
        return self.d.qpos[a : a + 3].copy(), self.d.qpos[a + 3 : a + 7].copy()

    def tile_rot(self, i: int) -> np.ndarray:
        return self.d.xmat[self.body[i]].reshape(3, 3).copy()

    def is_upright(self, i: int, tol_deg: float = 12.0) -> bool:
        return self.tile_rot(i)[:, 1][2] > np.cos(np.deg2rad(tol_deg))

    def is_face_up(self, i: int, tol_deg: float = 12.0) -> bool:
        return self.tile_rot(i)[:, 2][2] > np.cos(np.deg2rad(tol_deg))

    def in_gripper(self, i: int) -> bool:
        """Tile touches both pads."""
        pads = {self.m.geom("tip_left_pad_contact").id: 0, self.m.geom("tip_right_pad_contact").id: 1}
        seen = set()
        g = self.geom[i]
        for k in range(self.d.ncon):
            c = self.d.contact[k]
            if c.geom1 == g and c.geom2 in pads:
                seen.add(pads[c.geom2])
            elif c.geom2 == g and c.geom1 in pads:
                seen.add(pads[c.geom1])
        return len(seen) == 2
