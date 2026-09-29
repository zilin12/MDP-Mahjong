"""Win detection for the three winning structures, and shanten.

All functions take a 34-length count vector of the concealed hand and the number of
declared melds (Chow, Pung, Kong of any kind).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Sequence

from .tiles import TERMINALS_AND_HONORS

STANDARD = "standard"
SEVEN_PAIRS = "seven_pairs"
THIRTEEN_ORPHANS = "thirteen_orphans"


# ---------------------------------------------------------------- win detection


@lru_cache(maxsize=None)
def _suit_sets_only(c: tuple[int, ...]) -> bool:
    for i, n in enumerate(c):
        if n:
            break
    else:
        return True
    if n >= 3:
        rest = list(c)
        rest[i] -= 3
        if _suit_sets_only(tuple(rest)):
            return True
    if i <= 6 and c[i + 1] and c[i + 2]:
        rest = list(c)
        rest[i] -= 1
        rest[i + 1] -= 1
        rest[i + 2] -= 1
        if _suit_sets_only(tuple(rest)):
            return True
    return False


@lru_cache(maxsize=None)
def _suit_sets_and_pair(c: tuple[int, ...]) -> bool:
    for i, n in enumerate(c):
        if n >= 2:
            rest = list(c)
            rest[i] -= 2
            if _suit_sets_only(tuple(rest)):
                return True
    return False


def is_standard_win(counts: Sequence[int], n_melds: int) -> bool:
    if sum(counts) != 3 * (4 - n_melds) + 2:
        return False
    pair_seen = False
    for s in range(3):
        c = tuple(counts[9 * s : 9 * s + 9])
        r = sum(c) % 3
        if r == 1:
            return False
        if r == 2:
            if pair_seen or not _suit_sets_and_pair(c):
                return False
            pair_seen = True
        elif not _suit_sets_only(c):
            return False
    for t in range(27, 34):
        n = counts[t]
        if n == 2:
            if pair_seen:
                return False
            pair_seen = True
        elif n not in (0, 3):
            return False
    return pair_seen


def is_seven_pairs(counts: Sequence[int], n_melds: int) -> bool:
    # ADR 0001: four identical tiles count as two pairs; special hands need no melds.
    return n_melds == 0 and sum(counts) == 14 and all(n % 2 == 0 for n in counts)


def is_thirteen_orphans(counts: Sequence[int], n_melds: int) -> bool:
    if n_melds != 0 or sum(counts) != 14:
        return False
    if any(counts[t] == 0 for t in TERMINALS_AND_HONORS):
        return False
    return sum(counts[t] for t in TERMINALS_AND_HONORS) == 14


def winning_structure(counts: Sequence[int], n_melds: int) -> str | None:
    if is_standard_win(counts, n_melds):
        return STANDARD
    if is_seven_pairs(counts, n_melds):
        return SEVEN_PAIRS
    if is_thirteen_orphans(counts, n_melds):
        return THIRTEEN_ORPHANS
    return None


def is_win(counts: Sequence[int], n_melds: int) -> bool:
    return winning_structure(counts, n_melds) is not None


# ---------------------------------------------------------------- shanten


def _pareto(options: set[tuple[int, int, int]]) -> tuple[tuple[int, int, int], ...]:
    """Keep (melds, partials, pair) options not dominated component-wise."""
    opts = sorted(options, reverse=True)
    kept: list[tuple[int, int, int]] = []
    for o in opts:
        if not any(k[0] >= o[0] and k[1] >= o[1] and k[2] >= o[2] for k in kept):
            kept.append(o)
    return tuple(kept)


@lru_cache(maxsize=None)
def _suit_blocks(c: tuple[int, ...]) -> tuple[tuple[int, int, int], ...]:
    """Pareto set of (melds, partial sets, pair 0/1) decompositions of one suit."""
    for i, n in enumerate(c):
        if n:
            break
    else:
        return ((0, 0, 0),)
    results: set[tuple[int, int, int]] = set()

    def branch(deltas: tuple[tuple[int, int], ...], dm: int, dt: int, dp: int) -> None:
        rest = list(c)
        for j, k in deltas:
            rest[j] -= k
        for m, t, p in _suit_blocks(tuple(rest)):
            if p + dp <= 1:
                results.add((m + dm, t + dt, p + dp))

    if n >= 3:
        branch(((i, 3),), 1, 0, 0)
    if i <= 6 and c[i + 1] and c[i + 2]:
        branch(((i, 1), (i + 1, 1), (i + 2, 1)), 1, 0, 0)
    if n >= 2:
        branch(((i, 2),), 0, 0, 1)
        branch(((i, 2),), 0, 1, 0)
    if i <= 7 and c[i + 1]:
        branch(((i, 1), (i + 1, 1)), 0, 1, 0)
    if i <= 6 and c[i + 2]:
        branch(((i, 1), (i + 2, 1)), 0, 1, 0)
    branch(((i, 1),), 0, 0, 0)
    return _pareto(results)


@lru_cache(maxsize=None)
def _honor_blocks(c: tuple[int, ...]) -> tuple[tuple[int, int, int], ...]:
    m = sum(1 for n in c if n >= 3)
    pairs = sum(1 for n in c if n == 2)
    opts = {(m, pairs, 0)}
    if pairs:
        opts.add((m, pairs - 1, 1))
    return _pareto(opts)


def _merge(a, b):
    out = set()
    for m1, t1, p1 in a:
        for m2, t2, p2 in b:
            if p1 + p2 <= 1:
                out.add((m1 + m2, t1 + t2, p1 + p2))
    return _pareto(out)


def standard_shanten(counts: Sequence[int], n_melds: int) -> int:
    front = _suit_blocks(tuple(counts[0:9]))
    front = _merge(front, _suit_blocks(tuple(counts[9:18])))
    front = _merge(front, _suit_blocks(tuple(counts[18:27])))
    front = _merge(front, _honor_blocks(tuple(counts[27:34])))
    best = 8
    for m, t, p in front:
        m_total = min(4, m + n_melds)
        t_used = min(t, 4 - m_total)
        best = min(best, 8 - 2 * m_total - t_used - p)
    return best


def seven_pairs_shanten(counts: Sequence[int]) -> int:
    return 6 - min(7, sum(n // 2 for n in counts))


def thirteen_orphans_shanten(counts: Sequence[int]) -> int:
    kinds = sum(1 for t in TERMINALS_AND_HONORS if counts[t])
    pair = any(counts[t] >= 2 for t in TERMINALS_AND_HONORS)
    return 13 - kinds - (1 if pair else 0)


def shanten(counts: Sequence[int], n_melds: int) -> int:
    """Minimum over the three winning structures; -1 means a complete 14-tile hand."""
    s = standard_shanten(counts, n_melds)
    if n_melds == 0 and s > 0:
        s = min(s, seven_pairs_shanten(counts), thirteen_orphans_shanten(counts))
    if s == -1 or (s <= 0 and sum(counts) % 3 == 2):
        # The block formula can under-count for hands that hold four of a waiting tile;
        # a 14-tile hand is complete only if an explicit win check says so.
        return -1 if is_win(counts, n_melds) else max(s, 0)
    return s
