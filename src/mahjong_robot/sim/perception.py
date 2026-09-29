"""Tile detection and pose estimation from RGB-D frames (no simulator ground truth).

Overhead: back-project depth with the believed extrinsic, threshold height above the table,
split connected components, and fit a rectangle to each (PCA over the top-surface points).
Flat tiles are about 16 mm tall and upright tiles about 26 mm. A long upright component is
a close-packed row and is divided by the known tile width. Face crops are rectified with a
perspective warp to a canonical 40 x 52 image for the classifier.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from . import layout as L
from .cameras import Frame

CROP_W, CROP_H = 40, 52
FLAT_Z = (0.009, 0.021)  # top-surface height band for flat tiles
UPRIGHT_MIN_Z = 0.021
ROBOT_KEEPOUT_Y = 0.095  # the arm's own base occupies the table in front of y = 0.095


@dataclass
class Detection:
    pos: np.ndarray  # tile centre (m)
    yaw: float  # width-axis angle from world x, in (-pi/2, pi/2]
    upright: bool
    extent: tuple[float, float]  # measured (along width, across) in metres
    row_members: int = 1  # tiles in the component this came from
    crop: np.ndarray | None = None
    tile_type: int | None = None
    confidence: float = 0.0
    region: str = ""
    meta: dict = field(default_factory=dict)


def _wrap_half_pi(a: float) -> float:
    a = (a + np.pi / 2) % np.pi - np.pi / 2
    return a if a > -np.pi / 2 + 1e-9 else a + np.pi


def detect_tiles(frame: Frame, roi: tuple[float, float, float, float] | None = None) -> list[Detection]:
    """Detect tiles in an overhead frame. ``roi`` = (x0, x1, y0, y1) in world metres."""
    pts = frame.points()
    valid = frame.depth > 0
    z = pts[..., 2]
    x0, x1, y0, y1 = roi or (-L.TABLE_HALF_X, L.TABLE_HALF_X, ROBOT_KEEPOUT_Y, L.TABLE_Y1)
    inside = valid & (pts[..., 0] > x0) & (pts[..., 0] < x1) & (pts[..., 1] > y0) & (pts[..., 1] < y1)
    out: list[Detection] = []
    upright_mask = (inside & (z > UPRIGHT_MIN_Z) & (z < 0.034)).astype(np.uint8)
    flat_mask = (inside & (z > FLAT_Z[0]) & (z < FLAT_Z[1])).astype(np.uint8)
    # Upright tiles' side walls also fall in the flat band; remove pixels near upright tops.
    near_upright = cv2.dilate(upright_mask, np.ones((9, 9), np.uint8))
    flat_mask &= 1 - near_upright
    flat_mask = cv2.morphologyEx(flat_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    for mask, upright in ((upright_mask, True), (flat_mask, False)):
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=4)
        for k in range(1, n):
            if stats[k, cv2.CC_STAT_AREA] < 60:
                continue
            sel = labels == k
            p = pts[sel]
            out.extend(_fit_component(p, upright, frame))
    return out


def _row_boundaries(frame: Frame, c: np.ndarray, major: np.ndarray, lo: float, hi: float, n: int, top: float):
    """Refine the n-1 internal tile boundaries of an upright row from the dark rims on the tops.

    Samples image brightness along the row's centre line (0.25 mm steps) and takes the darkest
    point within +/-6 mm of each evenly spaced guess. Returns boundary offsets along ``major``.
    """
    s = np.arange(lo, hi, 0.00025)
    line = c[None, :] + s[:, None] * major[None, :]
    world = np.column_stack([line, np.full(len(s), top)])
    uv = frame.project(world)
    h, w = frame.depth.shape
    uv = np.clip(np.round(uv).astype(int), 0, [w - 1, h - 1])
    gray = frame.rgb[uv[:, 1], uv[:, 0]].astype(np.float32).mean(axis=1)
    gray = np.convolve(gray, np.ones(5) / 5, mode="same")
    bounds = []
    for k in range(1, n):
        guess = lo + k * (hi - lo) / n
        win = (s > guess - 0.006) & (s < guess + 0.006)
        if not win.any():
            bounds.append(guess)
            continue
        idx = np.flatnonzero(win)
        bounds.append(float(s[idx[np.argmin(gray[idx])]]))
    return bounds


def _fit_component(p: np.ndarray, upright: bool, frame: Frame | None = None) -> list[Detection]:
    xy = p[:, :2]
    c = xy.mean(axis=0)
    cov = np.cov((xy - c).T)
    w, v = np.linalg.eigh(cov)
    major = v[:, 1]
    proj_major = (xy - c) @ major
    proj_minor = (xy - c) @ v[:, 0]
    lo_M, hi_M = np.percentile(proj_major, [0.3, 99.7])
    lo_m, hi_m = np.percentile(proj_minor, [1.5, 98.5])
    len_M, len_m = hi_M - lo_M, hi_m - lo_m
    top = float(np.median(p[:, 2]))
    if upright:
        # Top face is W x T; a row is n*W x T. Width axis = major axis.
        n = max(1, int(round(len_M / L.TILE_W)))
        if len_M < 0.75 * L.TILE_W:
            return []
        yaw = _wrap_half_pi(np.arctan2(major[1], major[0]))
        edges = [lo_M] + (_row_boundaries(frame, c + v[:, 0] * (lo_m + hi_m) / 2, major, lo_M, hi_M, n, top)
                          if frame is not None and n > 1 else [lo_M + k * len_M / n for k in range(1, n)]) + [hi_M]
        dets = []
        for i in range(n):
            a, b_ = edges[i], edges[i + 1]
            # Each tile is W wide; anchor it on its better-measured edge when the span is off.
            off = (a + b_) / 2
            ci = c + major * off + v[:, 0] * (lo_m + hi_m) / 2
            dets.append(Detection(np.array([ci[0], ci[1], L.TILE_H / 2]), yaw, True,
                                  (b_ - a, len_m), row_members=n, meta={"top_z": top}))
        return dets
    # Flat face-up tiles: one tile is W x H. A merged blob is k tiles side by side (width along
    # the major axis) or end to end (height along the major axis); pick the better fit.
    if len_M < 0.7 * L.TILE_W or len_m < 0.6 * L.TILE_W:
        return []
    fits = []
    for along, across, width_major in ((L.TILE_W, L.TILE_H, True), (L.TILE_H, L.TILE_W, False)):
        n = max(1, int(round(len_M / (along + 0.003))))
        err = abs(len_M - n * along - (n - 1) * 0.003) + abs(len_m - across)
        fits.append((err, n, width_major))
    err, n, width_major = min(fits)
    wax = major if width_major else v[:, 0]
    yaw = _wrap_half_pi(np.arctan2(wax[1], wax[0]))
    dets = []
    for i in range(n):
        off = lo_M + (i + 0.5) * len_M / n
        ci = c + major * off + v[:, 0] * (lo_m + hi_m) / 2
        dets.append(Detection(np.array([ci[0], ci[1], L.TILE_T / 2]), yaw, False,
                              (len_M / n, len_m) if width_major else (len_m, len_M / n),
                              row_members=n, meta={"top_z": top, "fit_err": err}))
    return dets


def in_view(frame: Frame, corners_world: np.ndarray, margin: int = 2) -> bool:
    uv = frame.project(corners_world)
    h, w = frame.depth.shape
    in_front = np.all(((np.asarray(corners_world) - frame.t_wc) @ frame.R_wc)[:, 2] > 0.02)
    return bool(in_front and np.all(uv >= margin) and np.all(uv[:, 0] < w - margin) and np.all(uv[:, 1] < h - margin))


def _warp(frame: Frame, corners_world: np.ndarray) -> np.ndarray | None:
    """Corners ordered top-left, top-right, bottom-right, bottom-left of the face image."""
    if not in_view(frame, corners_world):
        return None
    src = frame.project(corners_world).astype(np.float32)
    dst = np.array([[0, 0], [CROP_W - 1, 0], [CROP_W - 1, CROP_H - 1], [0, CROP_H - 1]], np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(frame.rgb, M, (CROP_W, CROP_H), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)


def flat_face_corners(pos: np.ndarray, yaw: float) -> np.ndarray:
    """Top face corners of a face-up tile (face 'up' direction along +height axis)."""
    ex = np.array([np.cos(yaw), np.sin(yaw), 0.0])  # width axis
    ey = np.array([-np.sin(yaw), np.cos(yaw), 0.0])  # height axis
    c = np.array([pos[0], pos[1], L.TILE_T])
    hw, hh = L.TILE_W / 2, L.TILE_H / 2
    return np.array([c - hw * ex + hh * ey, c + hw * ex + hh * ey, c + hw * ex - hh * ey, c - hw * ex - hh * ey])


def upright_face_corners(pos: np.ndarray, yaw: float) -> np.ndarray:
    """Face corners of an upright tile whose face points toward the robot (-y rotated by yaw)."""
    ex = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    normal = np.array([np.sin(yaw), -np.cos(yaw), 0.0])
    c = np.array([pos[0], pos[1], 0.0]) + normal * (L.TILE_T / 2)
    hw = L.TILE_W / 2
    top, bot = np.array([0, 0, L.TILE_H]), np.zeros(3)
    return np.array([c - hw * ex + top, c + hw * ex + top, c + hw * ex + bot, c - hw * ex + bot])


def crop_flat(frame: Frame, det: Detection) -> np.ndarray | None:
    return _warp(frame, flat_face_corners(det.pos, det.yaw))


def crop_upright_face(frame: Frame, pos: np.ndarray, yaw: float) -> np.ndarray | None:
    return _warp(frame, upright_face_corners(pos, yaw))
