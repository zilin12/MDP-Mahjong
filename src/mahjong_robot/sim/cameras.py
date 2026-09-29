"""Simulated RGB-D cameras with calibration error and depth noise.

Images are rendered from the camera's true pose. The robot interprets them with its
*believed* pose: for the overhead D435i, a fixed extrinsic perturbed by a per-episode
calibration error; for the wrist D405, forward kinematics composed with a perturbed
hand-eye transform. Depth noise follows the stereo model sigma = z^2 * s / (f * b) with the
devices' published baselines and a 0.08 px subpixel error.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

CV_FROM_MJ = np.diag([1.0, -1.0, -1.0])  # MuJoCo camera axes -> OpenCV optical axes


@dataclass
class CameraSpec:
    name: str
    width: int
    height: int
    baseline: float  # stereo baseline (m)
    subpixel: float = 0.08
    min_depth: float = 0.07
    rgb_noise: float = 3.0


D435I = CameraSpec("overhead_d435i", 1280, 720, baseline=0.050)
D405 = CameraSpec("wrist_d405", 640, 360, baseline=0.018, min_depth=0.07)


def _small_rotation(rng: np.random.Generator, sigma_rad: float) -> np.ndarray:
    w = rng.normal(0.0, sigma_rad, 3)
    angle = np.linalg.norm(w)
    if angle < 1e-12:
        return np.eye(3)
    k = w / angle
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


@dataclass
class Frame:
    rgb: np.ndarray  # H x W x 3 uint8
    depth: np.ndarray  # H x W metres along the optical axis (0 = invalid)
    K: np.ndarray  # 3 x 3 intrinsics
    R_wc: np.ndarray  # believed camera-to-world rotation (OpenCV optical frame)
    t_wc: np.ndarray  # believed camera position in world
    time: float = 0.0
    meta: dict = field(default_factory=dict)

    def points(self) -> np.ndarray:
        """Back-project every pixel to world coordinates with the believed pose (H x W x 3)."""
        h, w = self.depth.shape
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        z = self.depth
        x = (u - self.K[0, 2]) * z / self.K[0, 0]
        y = (v - self.K[1, 2]) * z / self.K[1, 1]
        pc = np.stack([x, y, z], axis=-1)
        return pc @ self.R_wc.T + self.t_wc

    def project(self, pw: np.ndarray) -> np.ndarray:
        """World points (..., 3) -> pixel coordinates (..., 2) with the believed pose."""
        pc = (np.asarray(pw) - self.t_wc) @ self.R_wc
        uv = pc[..., :2] / pc[..., 2:3]
        return uv * np.array([self.K[0, 0], self.K[1, 1]]) + np.array([self.K[0, 2], self.K[1, 2]])


class SimCamera:
    """One device: renders from the true pose, reports the believed pose."""

    def __init__(self, model: mujoco.MjModel, spec: CameraSpec, rng: np.random.Generator,
                 calib_trans_sigma: float, calib_rot_sigma_deg: float):
        self.m, self.spec = model, spec
        self.cam_id = model.camera(spec.name).id
        self.renderer = mujoco.Renderer(model, spec.height, spec.width)
        self.depth_renderer = mujoco.Renderer(model, spec.height, spec.width)
        self.depth_renderer.enable_depth_rendering()
        fovy = np.deg2rad(model.cam_fovy[self.cam_id])
        f = (spec.height / 2) / np.tan(fovy / 2)
        self.K = np.array([[f, 0, (spec.width - 1) / 2], [0, f, (spec.height - 1) / 2], [0, 0, 1.0]])
        self.rng = rng
        # Calibration error, fixed for the episode (unknown to the robot).
        self.err_R = _small_rotation(rng, np.deg2rad(calib_rot_sigma_deg))
        self.err_t = rng.normal(0.0, calib_trans_sigma, 3)
        self.body = model.cam_bodyid[self.cam_id]

    def recalibrate(self, calib_trans_sigma: float, calib_rot_sigma_deg: float) -> None:
        """Draw a new calibration error (a new episode or session)."""
        self.err_R = _small_rotation(self.rng, np.deg2rad(calib_rot_sigma_deg))
        self.err_t = self.rng.normal(0.0, calib_trans_sigma, 3)

    def true_pose(self, d: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
        R = d.cam_xmat[self.cam_id].reshape(3, 3) @ CV_FROM_MJ
        return R, d.cam_xpos[self.cam_id].copy()

    def believed_pose(self, d: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
        """True pose seen through the calibration error, expressed in the mount frame."""
        R, t = self.true_pose(d)
        Rb, tb = d.xmat[self.body].reshape(3, 3), d.xpos[self.body]
        # Error acts in the mount (body) frame, like a mis-estimated extrinsic.
        R_rel = Rb.T @ R
        t_rel = Rb.T @ (t - tb)
        return Rb @ (self.err_R @ R_rel), tb + Rb @ (t_rel + self.err_t)

    def capture(self, d: mujoco.MjData) -> Frame:
        self.renderer.update_scene(d, self.cam_id)
        rgb = self.renderer.render().astype(np.float32)
        self.depth_renderer.update_scene(d, self.cam_id)
        depth = self.depth_renderer.render().astype(np.float64)
        sp = self.spec
        sigma = depth**2 * sp.subpixel / (self.K[0, 0] * sp.baseline)
        depth = depth + self.rng.normal(0.0, 1.0, depth.shape) * sigma
        depth[(depth < sp.min_depth) | (depth > 3.0)] = 0.0
        rgb = np.clip(rgb + self.rng.normal(0.0, sp.rgb_noise, rgb.shape), 0, 255).astype(np.uint8)
        R, t = self.believed_pose(d)
        return Frame(rgb, depth.astype(np.float32), self.K.copy(), R, t, float(d.time))

    def close(self) -> None:
        self.renderer.close()
        self.depth_renderer.close()
