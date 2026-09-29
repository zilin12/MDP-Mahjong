"""Fixed baseline players. Each acts only on its PlayerView.

- RandomPlayer: the legal random player.
- HeuristicPlayer: shanten-greedy training opponent (CONTEXT.md: Heuristic player).
- HeldOutHeuristicPlayer: a different fixed style used only for evaluation. It breaks
  discard ties by ukeire (unseen tiles that improve the hand) and never Chows.
"""

from __future__ import annotations

import random

from ..game import actions as A
from ..game.engine import PlayerView
from ..game.hand_eval import shanten
from ..game.tiles import N_TYPES, is_suited, rank_of, type_of


class Player:
    name = "player"

    def reset(self, seat: int, seed: int) -> None:
        self.seat = seat
        self.rng = random.Random(seed)

    def act(self, view: PlayerView) -> int:
        raise NotImplementedError


class RandomPlayer(Player):
    name = "random"

    def act(self, view: PlayerView) -> int:
        return self.rng.choice(view.legal)


def hand_counts(view: PlayerView) -> list[int]:
    c = [0] * N_TYPES
    for t in view.hand:
        c[type_of(t)] += 1
    return c


def visible_counts(view: PlayerView) -> list[int]:
    """Copies of each type this seat can see: own hand, all melds, unclaimed discards."""
    c = hand_counts(view)
    for melds in view.melds:
        for m in melds:
            for t in m.tiles:
                c[type_of(t)] += 1
    for pond in view.discards:
        for d in pond:
            if d.claimed_by is None:
                c[type_of(d.tile)] += 1
    return c


def isolation_score(counts: list[int], t: int) -> int:
    """How connected a tile is to the rest of the hand; lower means safer to discard."""
    if not is_suited(t):
        return 4 * (counts[t] - 1)
    score = 4 * (counts[t] - 1)
    r = rank_of(t)
    for d, w in ((-2, 1), (-1, 2), (1, 2), (2, 1)):
        if 1 <= r + d <= 9 and counts[t + d]:
            score += w
    return score + (0 if r in (1, 9) else 1)


class HeuristicPlayer(Player):
    name = "heuristic"
    chows = True

    def _discard_scores(self, counts, n_melds, view):
        """(shanten after discard, tie-break key) for each discardable type."""
        out = {}
        for t in range(N_TYPES):
            if counts[t]:
                counts[t] -= 1
                out[t] = (shanten(counts, n_melds), self._tiebreak(counts, t, n_melds, view))
                counts[t] += 1
        return out

    def _tiebreak(self, counts, t, n_melds, view) -> float:
        counts[t] += 1
        s = isolation_score(counts, t)
        counts[t] -= 1
        return s

    def act(self, view: PlayerView) -> int:
        legal = view.legal
        if A.SELF_DRAW_WIN in legal:
            return A.SELF_DRAW_WIN
        if A.WIN_DISCARD in legal:
            return A.WIN_DISCARD
        counts = hand_counts(view)
        n_melds = len(view.melds[view.seat])
        if view.phase == "claim":
            return self._claim(view, counts, n_melds)
        scores = self._discard_scores(counts, n_melds, view)
        best = min(s for s, _ in scores.values())
        for a in legal:
            if A.CONCEALED_KONG <= a < A.ADDED_KONG:
                t = a - A.CONCEALED_KONG
                counts[t] -= 4
                ok = shanten(counts, n_melds + 1) <= best
                counts[t] += 4
                if ok:
                    return a
            elif A.ADDED_KONG <= a < A.SELF_DRAW_WIN and scores[a - A.ADDED_KONG][0] <= best:
                return a
        options = [t for t, (s, _) in scores.items() if s == best]
        low = min(scores[t][1] for t in options)
        options = [t for t in options if scores[t][1] == low]
        return A.discard(self.rng.choice(options))

    def _claim(self, view, counts, n_melds) -> int:
        legal = view.legal
        t = type_of(view.current_discard.tile)
        now = shanten(counts, n_melds)
        best_action, best_value = A.PASS, now
        for a in legal:
            if a == A.KONG:
                counts[t] -= 3
                if shanten(counts, n_melds + 1) <= now:
                    best_action, best_value = a, now - 0.5
                counts[t] += 3
                continue
            if a == A.PUNG:
                used = (t, t)
            elif a in A.CHOW_OFFSETS and self.chows:
                used = tuple(t + d for d in A.CHOW_OFFSETS[a])
            else:
                continue
            for u in used:
                counts[u] -= 1
            after = min(
                (self._after_discard(counts, n_melds + 1, d) for d in range(N_TYPES) if counts[d]), default=9
            )
            for u in used:
                counts[u] += 1
            if after < best_value:
                best_action, best_value = a, after
        return best_action

    @staticmethod
    def _after_discard(counts, n_melds, d) -> int:
        counts[d] -= 1
        s = shanten(counts, n_melds)
        counts[d] += 1
        return s


class HeldOutHeuristicPlayer(HeuristicPlayer):
    name = "heldout_heuristic"
    chows = False

    def _tiebreak(self, counts, t, n_melds, view) -> float:
        # More unseen improving tiles is better, so negate for the min-selection.
        seen = visible_counts(view)
        base = shanten(counts, n_melds)
        ukeire = 0
        for d in _candidate_draws(counts):
            if seen[d] >= 4:
                continue
            counts[d] += 1
            if shanten(counts, n_melds) < base:
                ukeire += 4 - seen[d]
            counts[d] -= 1
        return -ukeire


def _candidate_draws(counts: list[int]) -> list[int]:
    out = set()
    for t in range(N_TYPES):
        if not counts[t]:
            continue
        out.add(t)
        if is_suited(t):
            r = rank_of(t)
            for d in (-2, -1, 1, 2):
                if 1 <= r + d <= 9:
                    out.add(t + d)
    out.update((0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33))
    return sorted(out)


PLAYERS = {cls.name: cls for cls in (RandomPlayer, HeuristicPlayer, HeldOutHeuristicPlayer)}
