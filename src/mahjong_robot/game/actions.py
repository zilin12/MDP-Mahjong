"""Semantic action space: 110 actions named by tile type (ADR 0002)."""

from __future__ import annotations

from .tiles import N_TYPES, name

DISCARD = 0  # + tile type
CONCEALED_KONG = DISCARD + N_TYPES  # + tile type
ADDED_KONG = CONCEALED_KONG + N_TYPES  # + tile type
SELF_DRAW_WIN = ADDED_KONG + N_TYPES  # 102
PASS = SELF_DRAW_WIN + 1
WIN_DISCARD = PASS + 1
PUNG = WIN_DISCARD + 1
KONG = PUNG + 1  # exposed Kong from a discard
CHOW_LOW = KONG + 1  # discard is the lowest tile of the sequence
CHOW_MID = CHOW_LOW + 1
CHOW_HIGH = CHOW_MID + 1
N_ACTIONS = CHOW_HIGH + 1

CHOW_OFFSETS = {CHOW_LOW: (1, 2), CHOW_MID: (-1, 1), CHOW_HIGH: (-2, -1)}
CLAIM_ACTIONS = (PASS, WIN_DISCARD, PUNG, KONG, CHOW_LOW, CHOW_MID, CHOW_HIGH)


def discard(tile_type: int) -> int:
    return DISCARD + tile_type


def concealed_kong(tile_type: int) -> int:
    return CONCEALED_KONG + tile_type


def added_kong(tile_type: int) -> int:
    return ADDED_KONG + tile_type


def tile_of(action: int) -> int | None:
    """Tile type carried by a turn action, or None for claim actions."""
    return action % N_TYPES if action < SELF_DRAW_WIN else None


def describe(action: int) -> str:
    if action < CONCEALED_KONG:
        return f"discard {name(action)}"
    if action < ADDED_KONG:
        return f"concealed kong {name(action - CONCEALED_KONG)}"
    if action < SELF_DRAW_WIN:
        return f"added kong {name(action - ADDED_KONG)}"
    return {
        SELF_DRAW_WIN: "self-draw win",
        PASS: "pass",
        WIN_DISCARD: "win by discard",
        PUNG: "pung",
        KONG: "kong",
        CHOW_LOW: "chow (discard low)",
        CHOW_MID: "chow (discard middle)",
        CHOW_HIGH: "chow (discard high)",
    }[action]
