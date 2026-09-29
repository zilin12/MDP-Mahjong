"""Encode a PlayerView as the policy's observation vector and legal-action mask.

Only PlayerView fields are read, so hidden-information isolation holds by construction.
Seats are expressed relative to the observing seat (0 = self, 1 = next, ...).
"""

from __future__ import annotations

import numpy as np

from ..game import actions as A
from ..game.engine import AFTER_CALL, CLAIM, TURN, PlayerView
from ..game.hand_eval import shanten
from ..game.tiles import N_TYPES, type_of
from .baselines import hand_counts

PHASES = (TURN, AFTER_CALL, CLAIM)
CLAIM_FEATURE_ACTIONS = (A.PUNG, A.KONG, A.CHOW_LOW, A.CHOW_MID, A.CHOW_HIGH, A.WIN_DISCARD)

_SIZES = {
    "hand_planes": 4 * N_TYPES,
    "melds": 4 * N_TYPES,
    "ponds": 4 * N_TYPES,
    "unseen": N_TYPES,
    "current_discard": N_TYPES,
    "discarder": 4,
    "last_draw": N_TYPES,
    "phase": len(PHASES),
    "scalars": 1 + 4 + 4,
    "shanten_now": 10,
    "shanten_after_discard": N_TYPES,
    "shanten_after_claim": len(CLAIM_FEATURE_ACTIONS),
}
OBS_DIM = sum(_SIZES.values())


def _best_discard_shanten(counts: list[int], n_melds: int) -> int:
    best = 9
    for d in range(N_TYPES):
        if counts[d]:
            counts[d] -= 1
            best = min(best, shanten(counts, n_melds))
            counts[d] += 1
    return best


def encode(view: PlayerView) -> tuple[np.ndarray, np.ndarray]:
    seat = view.seat
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    counts = hand_counts(view)
    n_melds = len(view.melds[seat])
    o = 0

    for t, n in enumerate(counts):
        for k in range(n):
            obs[o + k * N_TYPES + t] = 1.0
    o += 4 * N_TYPES

    visible = list(counts)
    for s in range(4):
        rel = (s - seat) % 4
        for m in view.melds[s]:
            for tile in m.tiles:
                tt = type_of(tile)
                obs[o + rel * N_TYPES + tt] += 0.25
                visible[tt] += 1
    o += 4 * N_TYPES

    for s in range(4):
        rel = (s - seat) % 4
        for d in view.discards[s]:
            if d.claimed_by is None:
                tt = type_of(d.tile)
                obs[o + rel * N_TYPES + tt] += 0.25
                visible[tt] += 1
    o += 4 * N_TYPES

    for t in range(N_TYPES):
        obs[o + t] = (4 - min(4, visible[t])) / 4.0
    o += N_TYPES

    cur = view.current_discard
    if cur is not None and view.phase == CLAIM:
        obs[o + type_of(cur.tile)] = 1.0
        obs[o + N_TYPES + (cur.seat - seat) % 4] = 1.0
    o += N_TYPES + 4

    if view.last_draw is not None:
        obs[o + type_of(view.last_draw)] = 1.0
    o += N_TYPES

    obs[o + PHASES.index(view.phase)] = 1.0
    o += len(PHASES)

    obs[o] = view.wall_remaining / 83.0
    obs[o + 1 + seat] = 1.0
    for s in range(4):
        obs[o + 5 + (s - seat) % 4] = len(view.melds[s]) / 4.0
    o += 9

    if view.phase == CLAIM:
        now = shanten(counts, n_melds)
    else:
        now = _best_discard_shanten(counts, n_melds)
        if A.SELF_DRAW_WIN in view.legal:
            now = -1
    obs[o + min(9, now + 1)] = 1.0
    o += 10

    if view.phase != CLAIM:
        for t in range(N_TYPES):
            if counts[t]:
                counts[t] -= 1
                obs[o + t] = 1.0 - (shanten(counts, n_melds) + 1) / 9.0
                counts[t] += 1
    o += N_TYPES

    if view.phase == CLAIM and cur is not None:
        t = type_of(cur.tile)
        for i, a in enumerate(CLAIM_FEATURE_ACTIONS):
            if a not in view.legal:
                continue
            if a == A.WIN_DISCARD:
                obs[o + i] = 1.0
                continue
            used = {A.PUNG: (t, t), A.KONG: (t, t, t)}.get(a) or tuple(t + d for d in A.CHOW_OFFSETS[a])
            for u in used:
                counts[u] -= 1
            s_after = shanten(counts, n_melds + 1) if a == A.KONG else _best_discard_shanten(counts, n_melds + 1)
            for u in used:
                counts[u] += 1
            obs[o + i] = 1.0 - (s_after + 1) / 9.0
    o += len(CLAIM_FEATURE_ACTIONS)
    assert o == OBS_DIM

    mask = np.zeros(A.N_ACTIONS, dtype=bool)
    mask[list(view.legal)] = True
    return obs, mask


def potential(view: PlayerView) -> float:
    """Shaping potential: minus the best reachable shanten, scaled to [-1, 0]."""
    counts = hand_counts(view)
    n_melds = len(view.melds[view.seat])
    s = shanten(counts, n_melds) if view.phase == CLAIM else _best_discard_shanten(counts, n_melds)
    return -max(0, s) / 8.0
