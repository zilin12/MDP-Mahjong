"""Rules engine for one hand under the fixed rule profile (docs/adr/0001-rule-profile.md).

The engine owns every concealed hand and the wall order. Seats see the game only through
``view(seat)``. Decisions are sequential: at a claim window each seat with a non-pass
option is asked in turn order, and no seat sees another's claim before resolution, so this
is equivalent to simultaneous claims. Forced draws happen inside ``step``.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from . import actions as A
from .hand_eval import winning_structure
from .tiles import COPIES, N_INSTANCES, N_TYPES, is_suited, rank_of, type_of

TURN = "turn"  # after a draw: win, Kong, or discard
AFTER_CALL = "after_call"  # after a Chow or Pung: discard only
CLAIM = "claim"  # claim window on the current discard
OVER = "over"

N_SEATS = 4
HAND_SIZE = 13


class IllegalAction(ValueError):
    pass


@dataclass(frozen=True)
class Meld:
    kind: str  # "chow", "pung", "kong_exposed", "kong_concealed", "kong_added"
    tiles: tuple[int, ...]  # tile instances
    claimed: int | None = None  # the claimed discard instance
    from_seat: int | None = None

    @property
    def tile_type(self) -> int:
        """Lowest tile type in the meld."""
        return min(type_of(t) for t in self.tiles)

    @property
    def is_kong(self) -> bool:
        return self.kind.startswith("kong")


def turn_actions(counts: list[int], melds: list[Meld], wall_remaining: int, phase: str) -> tuple[int, ...]:
    """Legal actions of the seat to act in TURN or AFTER_CALL (ADR 0001 rules 4-6)."""
    acts = [A.DISCARD + t for t in range(N_TYPES) if counts[t]]
    if phase == TURN:
        if wall_remaining > 0:
            acts += [A.CONCEALED_KONG + t for t in range(N_TYPES) if counts[t] == 4]
            acts += [A.ADDED_KONG + m.tile_type for m in melds if m.kind == "pung" and counts[m.tile_type]]
        if winning_structure(counts, len(melds)):
            acts.append(A.SELF_DRAW_WIN)
    return tuple(acts)


def claim_options(counts: list[int], n_melds: int, t: int, is_next_seat: bool, wall_remaining: int) -> tuple[int, ...]:
    """Legal claims on a discard of type ``t`` (ADR 0001 rules 8 and 12)."""
    opts = [A.PASS]
    counts = list(counts)
    counts[t] += 1
    if winning_structure(counts, n_melds):
        opts.append(A.WIN_DISCARD)
    counts[t] -= 1
    if counts[t] >= 2:
        opts.append(A.PUNG)
    if counts[t] == 3 and wall_remaining > 0:
        opts.append(A.KONG)
    if is_next_seat and is_suited(t):
        r = rank_of(t)
        for act, (d1, d2) in A.CHOW_OFFSETS.items():
            if 1 <= r + d1 <= 9 and 1 <= r + d2 <= 9 and counts[t + d1] and counts[t + d2]:
                opts.append(act)
    return tuple(opts)


@dataclass(frozen=True)
class Result:
    winner: int | None  # None on wall exhaustion
    structure: str | None = None
    win_type: str | None = None  # "self_draw" or "discard"
    from_seat: int | None = None  # the discarder on a win by discard
    winning_tile: int | None = None


@dataclass(frozen=True)
class DiscardRecord:
    seat: int
    tile: int
    claimed_by: int | None = None


@dataclass(frozen=True)
class PlayerView:
    """Everything one seat may know (CONTEXT.md: Player view)."""

    seat: int
    phase: str
    to_act: int
    legal: tuple[int, ...]  # empty unless this seat is to act
    hand: tuple[int, ...]  # own concealed tile instances
    last_draw: int | None  # own last drawn instance while it is this seat's turn
    melds: tuple[tuple[Meld, ...], ...]  # per seat, public
    discards: tuple[tuple[DiscardRecord, ...], ...]  # per seat, public, in order
    current_discard: DiscardRecord | None
    wall_remaining: int
    concealed_counts: tuple[int, ...]  # number of concealed tiles per seat (public)
    result: Result | None


@dataclass
class _ClaimWindow:
    discard: DiscardRecord
    queue: list[tuple[int, tuple[int, ...]]]  # (seat, legal claim actions) still to ask
    claims: dict[int, int] = field(default_factory=dict)


class Engine:
    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)
        self.reset(seed)

    # ------------------------------------------------------------ setup

    def reset(self, seed: int | None = None, wall: list[int] | None = None) -> None:
        if seed is not None:
            self.rng = random.Random(seed)
        if wall is None:
            wall = list(range(N_INSTANCES))
            self.rng.shuffle(wall)
        if sorted(wall) != list(range(N_INSTANCES)):
            raise ValueError("wall must be a permutation of the 136 tile instances")
        self.wall = list(wall)
        self.front = 0
        self.back = N_INSTANCES
        self.hands: list[list[int]] = [[] for _ in range(N_SEATS)]
        self.counts: list[list[int]] = [[0] * N_TYPES for _ in range(N_SEATS)]
        self.melds: list[list[Meld]] = [[] for _ in range(N_SEATS)]
        self.discards: list[list[DiscardRecord]] = [[] for _ in range(N_SEATS)]
        self.events: list[tuple] = []
        self.result: Result | None = None
        self.last_draw: int | None = None
        self.window: _ClaimWindow | None = None
        for seat in range(N_SEATS):
            for _ in range(HAND_SIZE):
                self._add(seat, self._pop_front())
        self.events.append(("deal",))
        self.to_act = 0
        self._draw(0, from_back=False, label="deal_draw")

    # ------------------------------------------------------------ tile moves

    def _pop_front(self) -> int:
        tile = self.wall[self.front]
        self.front += 1
        return tile

    def _pop_back(self) -> int:
        self.back -= 1
        return self.wall[self.back]

    @property
    def wall_remaining(self) -> int:
        return self.back - self.front

    def _add(self, seat: int, tile: int) -> None:
        self.hands[seat].append(tile)
        self.counts[seat][type_of(tile)] += 1

    def _remove_type(self, seat: int, tile_type: int, prefer: int | None = None) -> int:
        """Remove one instance of ``tile_type`` from a hand and return it."""
        hand = self.hands[seat]
        if prefer is not None:
            if prefer not in hand or type_of(prefer) != tile_type:
                raise IllegalAction(f"instance {prefer} is not a {tile_type} in seat {seat}'s hand")
            tile = prefer
        elif self.last_draw is not None and type_of(self.last_draw) == tile_type and self.last_draw in hand:
            tile = self.last_draw
        else:
            tile = max(t for t in hand if type_of(t) == tile_type)
        hand.remove(tile)
        self.counts[seat][tile_type] -= 1
        return tile

    def _draw(self, seat: int, from_back: bool, label: str | None = None) -> None:
        self.to_act = seat
        label = label or ("replacement_draw" if from_back else "draw")
        tile = self._pop_back() if from_back else self._pop_front()
        self._add(seat, tile)
        self.last_draw = tile
        self.events.append((label, seat, tile))
        self.phase = TURN

    # ------------------------------------------------------------ legality

    @property
    def done(self) -> bool:
        return self.phase == OVER

    def n_melds(self, seat: int) -> int:
        return len(self.melds[seat])

    def legal_actions(self) -> tuple[int, ...]:
        if self.phase == OVER:
            return ()
        if self.phase == CLAIM:
            return self.window.queue[0][1]
        seat = self.to_act
        return turn_actions(self.counts[seat], self.melds[seat], self.wall_remaining, self.phase)

    def _claim_options(self, seat: int, discard: DiscardRecord) -> tuple[int, ...]:
        return claim_options(self.counts[seat], self.n_melds(seat), type_of(discard.tile),
                             seat == (discard.seat + 1) % N_SEATS, self.wall_remaining)

    # ------------------------------------------------------------ transitions

    def step(self, action: int, instance: int | None = None) -> None:
        """Apply the acting seat's action. ``instance`` optionally binds the tile instance."""
        legal = self.legal_actions()
        if action not in legal:
            raise IllegalAction(f"{A.describe(action)} is not legal for seat {self.to_act} in {self.phase}")
        if self.phase == CLAIM:
            self._claim(action)
        elif action == A.SELF_DRAW_WIN:
            seat = self.to_act
            self.result = Result(
                winner=seat,
                structure=winning_structure(self.counts[seat], self.n_melds(seat)),
                win_type="self_draw",
                winning_tile=self.last_draw,
            )
            self.events.append(("win", seat, "self_draw", self.last_draw))
            self.phase = OVER
        elif action < A.CONCEALED_KONG:
            self._discard(action - A.DISCARD, instance)
        elif action < A.ADDED_KONG:
            seat, t = self.to_act, action - A.CONCEALED_KONG
            tiles = tuple(self._remove_type(seat, t) for _ in range(COPIES))
            self.melds[seat].append(Meld("kong_concealed", tiles))
            self.events.append(("kong_concealed", seat, tiles))
            self._draw(seat, from_back=True)
        else:
            seat, t = self.to_act, action - A.ADDED_KONG
            idx = next(i for i, m in enumerate(self.melds[seat]) if m.kind == "pung" and m.tile_type == t)
            old = self.melds[seat][idx]
            tile = self._remove_type(seat, t, instance)
            self.melds[seat][idx] = Meld("kong_added", old.tiles + (tile,), old.claimed, old.from_seat)
            self.events.append(("kong_added", seat, tile))
            self._draw(seat, from_back=True)

    def _discard(self, tile_type: int, instance: int | None) -> None:
        seat = self.to_act
        tile = self._remove_type(seat, tile_type, instance)
        self.last_draw = None
        record = DiscardRecord(seat, tile)
        self.discards[seat].append(record)
        self.events.append(("discard", seat, tile))
        queue = []
        for k in range(1, N_SEATS):
            other = (seat + k) % N_SEATS
            opts = self._claim_options(other, record)
            if len(opts) > 1:
                queue.append((other, opts))
        self.window = _ClaimWindow(record, queue)
        self.phase = CLAIM
        if queue:
            self.to_act = queue[0][0]
        else:
            self._resolve_claims()

    def _claim(self, action: int) -> None:
        seat, _ = self.window.queue.pop(0)
        self.window.claims[seat] = action
        if self.window.queue:
            self.to_act = self.window.queue[0][0]
        else:
            self._resolve_claims()

    def _resolve_claims(self) -> None:
        window = self.window
        discard = window.discard
        t = type_of(discard.tile)
        order = [(discard.seat + k) % N_SEATS for k in range(1, N_SEATS)]
        claims = window.claims
        self.window = None

        winners = [s for s in order if claims.get(s) == A.WIN_DISCARD]
        if winners:
            seat = winners[0]  # nearest single winner (ADR 0001, rule 8)
            self._mark_claimed(discard, seat)
            self._add(seat, discard.tile)
            self.result = Result(
                winner=seat,
                structure=winning_structure(self.counts[seat], self.n_melds(seat)),
                win_type="discard",
                from_seat=discard.seat,
                winning_tile=discard.tile,
            )
            self.events.append(("win", seat, "discard", discard.tile, discard.seat))
            self.phase = OVER
            return

        for seat in order:
            act = claims.get(seat)
            if act in (A.PUNG, A.KONG):
                self._mark_claimed(discard, seat)
                n = 2 if act == A.PUNG else 3
                own = tuple(self._remove_type(seat, t) for _ in range(n))
                kind = "pung" if act == A.PUNG else "kong_exposed"
                self.melds[seat].append(Meld(kind, own + (discard.tile,), discard.tile, discard.seat))
                self.events.append((kind, seat, discard.tile, discard.seat, own))
                if act == A.PUNG:
                    self.to_act, self.phase, self.last_draw = seat, AFTER_CALL, None
                else:
                    self._draw(seat, from_back=True)
                return

        chower = order[0]
        act = claims.get(chower)
        if act in A.CHOW_OFFSETS:
            self._mark_claimed(discard, chower)
            own = tuple(self._remove_type(chower, t + d) for d in A.CHOW_OFFSETS[act])
            self.melds[chower].append(Meld("chow", own + (discard.tile,), discard.tile, discard.seat))
            self.events.append(("chow", chower, discard.tile, discard.seat, own))
            self.to_act, self.phase, self.last_draw = chower, AFTER_CALL, None
            return

        nxt = (discard.seat + 1) % N_SEATS
        if self.wall_remaining == 0:
            self.result = Result(winner=None)
            self.events.append(("exhausted",))
            self.phase = OVER
            return
        self._draw(nxt, from_back=False)

    def _mark_claimed(self, discard: DiscardRecord, seat: int) -> None:
        pond = self.discards[discard.seat]
        pond[-1] = DiscardRecord(discard.seat, discard.tile, claimed_by=seat)

    # ------------------------------------------------------------ views

    def view(self, seat: int) -> PlayerView:
        acting = self.phase != OVER and self.to_act == seat
        cur = None
        if self.phase == CLAIM:
            cur = self.window.discard
        return PlayerView(
            seat=seat,
            phase=self.phase,
            to_act=self.to_act,
            legal=self.legal_actions() if acting else (),
            hand=tuple(sorted(self.hands[seat])),
            last_draw=self.last_draw if self.to_act == seat and self.phase == TURN else None,
            melds=tuple(tuple(m) for m in self.melds),
            discards=tuple(tuple(d) for d in self.discards),
            current_discard=cur,
            wall_remaining=self.wall_remaining,
            concealed_counts=tuple(len(h) for h in self.hands),
            result=self.result,
        )

    def public_events(self, seat: int) -> list[tuple]:
        """Event log with other seats' draw identities removed."""
        out = []
        for e in self.events:
            if e[0] in ("draw", "deal_draw", "replacement_draw") and e[1] != seat:
                out.append((e[0], e[1]))
            else:
                out.append(e)
        return out

    def tile_census(self) -> list[int]:
        """All instances in wall, hands, melds, and unclaimed discards (for conservation checks)."""
        tiles = list(self.wall[self.front : self.back])
        for s in range(N_SEATS):
            tiles += self.hands[s]
            for m in self.melds[s]:
                tiles += list(m.tiles)
            tiles += [d.tile for d in self.discards[s] if d.claimed_by is None]
        return tiles
