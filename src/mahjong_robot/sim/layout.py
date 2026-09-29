"""Table geometry in the robot base frame (metres).

x points right, y points away from the robot across the table, z up; z = 0 is the table
top and the YAM base sits at the origin. Tile body frame: x = width, y = height, z =
thickness, with the engraved face on +z. A flat face-up tile uses the identity orientation;
an upright hand tile is rotated +90 degrees about x, so its face points at the robot (-y).
"""

from __future__ import annotations

import numpy as np

TILE_W, TILE_H, TILE_T = 0.020, 0.026, 0.016
TILE_HALF = np.array([TILE_W, TILE_H, TILE_T]) / 2
TILE_MASS = 0.012

TABLE_HALF_X, TABLE_Y0, TABLE_Y1 = 0.45, -0.08, 0.90

# Close-packed hand row: upright tiles touching along x, dealt with the right end at
# HAND_ANCHOR_X. The row may drift within [HAND_X_MIN, HAND_X_MAX] (tile centres), the span
# where an upright tile can be held face-preserving: new tiles go to the end with more room
# and compaction pushes the smaller outer segment toward the gap.
HAND_Y = 0.170
HAND_X0 = -0.170
HAND_ANCHOR_X = HAND_X0 + 12.5 * TILE_W
HAND_X_MIN, HAND_X_MAX = -0.190, 0.100
INSERT_MARGIN = 0.0025  # outward clearance when inserting at a free end

# Robot's discard pond: 3 rows x 8 face-up slots, filled far row first so a tile standing
# before its tip-over (ADR 0004) never overlaps an earlier row. Slots the robot fills are
# 40 mm apart across a row so the side-grasp pads (12 mm thick, opened 4 mm) clear the
# neighbouring tile. Rows stay within y <= 0.30 m: beyond that the tip-over's roll needs
# more than the YAM's 10 N m joint torque.
POND_X0, POND_DX, POND_COLS = -0.140, 0.040, 8
POND_ROWS_Y = (0.300, 0.265, 0.230)

# Robot's exposed melds: four reserved slots per meld (an added Kong fills the fourth),
# a block right of the hand then a strip between the hand and the robot, far rows first.
MELD_SLOTS = (
    [(0.18, 0.25), (0.22, 0.25), (0.18, 0.21), (0.22, 0.21),
     (0.26, 0.21), (0.18, 0.17), (0.22, 0.17), (0.26, 0.17)]
    + [(-0.14, 0.125), (-0.10, 0.125), (-0.06, 0.125), (-0.02, 0.125), (0.10, 0.125), (0.14, 0.125),
       (0.18, 0.13), (0.22, 0.13)]
)

# Staging spot where a tile is stood upright and regrasped by its sides before a tip-over.
# Holding an upright tile face-preserving needs joint 6 within its limits, which restricts
# the reachable bearing (from +x) to about 60-150 degrees; the side grasp has no such limit.
STAGING_XY = np.array([-0.200, 0.260])

# Shared zone where the table (simulated human) presents tiles for the robot to take.
SHARED_XY = np.array([-0.260, 0.200])
SHARED_JITTER_XY, SHARED_JITTER_YAW = 0.006, np.deg2rad(8.0)

# Opponent ponds (relative seat 1 = right, 2 = across, 3 = left) and meld strips.
# Kept within x = +/-0.28 m so every tile is fully inside the overhead image.
OPP_POND = {1: (0.150, 0.440), 2: (-0.056, 0.640), 3: (-0.262, 0.440)}
OPP_MELD = {1: (0.150, 0.600), 2: (-0.056, 0.800), 3: (-0.262, 0.600)}
OPP_COLS = 5
# Humans place opponents' discards 28 x 32 mm apart (gaps keep them separable in depth).
OPP_DX, OPP_DY = 0.028, 0.032

# Arm pose used for overhead captures: parked left and low, out of the camera's view of the work areas.
PARK_XYZ = np.array([-0.30, 0.02, 0.22])

# Where unused tiles float (contacts disabled, gravity compensated) off the table.
STORAGE_ORIGIN = np.array([1.5, 0.0, 0.3])

UPRIGHT_QUAT = np.array([np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0, 0.0])  # +90 deg about x
FLAT_QUAT = np.array([1.0, 0.0, 0.0, 0.0])


def hand_slot_x(i: int) -> float:
    return HAND_X0 + (i + 0.5) * TILE_W


def hand_pos(i: int) -> np.ndarray:
    return np.array([hand_slot_x(i), HAND_Y, TILE_H / 2])


def pond_pos(k: int) -> np.ndarray:
    row, col = divmod(k % (POND_COLS * len(POND_ROWS_Y)), POND_COLS)
    return np.array([POND_X0 + col * POND_DX, POND_ROWS_Y[row], TILE_T / 2])


def meld_pos(meld_index: int, tile_index: int) -> np.ndarray:
    x, y = MELD_SLOTS[4 * meld_index + tile_index]
    return np.array([x, y, TILE_T / 2])


def opp_pond_pos(rel_seat: int, k: int) -> np.ndarray:
    x0, y0 = OPP_POND[rel_seat]
    return np.array([x0 + (k % OPP_COLS) * OPP_DX, y0 + (k // OPP_COLS) * OPP_DY, TILE_T / 2])


def opp_meld_pos(rel_seat: int, k: int) -> np.ndarray:
    x0, y0 = OPP_MELD[rel_seat]
    return np.array([x0 + (k % OPP_COLS) * OPP_DX, y0 + (k // OPP_COLS) * 0.034, TILE_T / 2])


def storage_pos(instance: int) -> np.ndarray:
    return STORAGE_ORIGIN + np.array([0.05 * (instance % 12), 0.05 * (instance // 12), 0.0])


def yaw_quat(yaw: float, base: np.ndarray = FLAT_QUAT) -> np.ndarray:
    """World yaw rotation composed with a base orientation (w, x, y, z)."""
    qz = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
    return quat_mul(qz, base)


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])
