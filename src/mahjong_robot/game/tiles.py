"""Tile types and tile instances.

Tile types are 0-33: Characters 1-9 (0-8), Bamboos 1-9 (9-17), Dots 1-9 (18-26),
Winds East/South/West/North (27-30), Dragons Red/Green/White (31-33).
Tile instances are 0-135; instance ``i`` has type ``i // 4``.
"""

from __future__ import annotations

N_TYPES = 34
N_INSTANCES = 136
COPIES = 4

SUIT_NAMES = ("C", "B", "D")
HONOR_NAMES = ("E", "S", "W", "N", "RD", "GD", "WD")
TYPE_NAMES = tuple(f"{r}{s}" for s in SUIT_NAMES for r in range(1, 10)) + HONOR_NAMES
LONG_NAMES = (
    tuple(f"{r} {s}" for s in ("Character", "Bamboo", "Dot") for r in range(1, 10))
    + ("East", "South", "West", "North", "Red Dragon", "Green Dragon", "White Dragon")
)
TERMINALS_AND_HONORS = (0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33)


def type_of(instance: int) -> int:
    return instance // COPIES


def instances_of(tile_type: int) -> range:
    return range(tile_type * COPIES, tile_type * COPIES + COPIES)


def is_suited(tile_type: int) -> bool:
    return tile_type < 27


def suit_of(tile_type: int) -> int:
    """0-2 for suited tiles, 3 for honors."""
    return tile_type // 9 if tile_type < 27 else 3


def rank_of(tile_type: int) -> int:
    """1-9 for suited tiles."""
    return tile_type % 9 + 1


def name(tile_type: int) -> str:
    return TYPE_NAMES[tile_type]


def parse(text: str) -> list[int]:
    """Parse a compact hand string such as ``"123C 456B 99D E E RD"`` into tile types.

    Digits before a suit letter share that suit; honors are named tokens.
    """
    out: list[int] = []
    for token in text.split():
        if token in HONOR_NAMES:
            out.append(27 + HONOR_NAMES.index(token))
            continue
        suit = token[-1]
        if suit not in SUIT_NAMES or not token[:-1].isdigit():
            raise ValueError(f"cannot parse tile token {token!r}")
        base = 9 * SUIT_NAMES.index(suit)
        out.extend(base + int(d) - 1 for d in token[:-1])
    return out


def counts_of(types: list[int]) -> list[int]:
    c = [0] * N_TYPES
    for t in types:
        c[t] += 1
    return c
