"""The simulated table: rules engine, the three other seats, and the humans' role (ADR 0003).

The table owns the rules engine and the physical tiles. It plays the three other seats, deals,
presents the robot's draws upright in the shared zone, moves claimed discards to the shared
zone face-up, and lays out the other seats' discards and melds. It sends the robot event
signals that carry timing only (``Signal``), never tile identity. Physical tiles are matched
to engine tiles by type: whenever the engine moves a tile of type t onto the table, the table
places any unplaced physical instance of type t.

The robot reports each semantic action only after executing and verifying it. ``commit``
applies it to the engine; the evaluator's ground-truth checks live here too.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..game import actions as A
from ..game.engine import AFTER_CALL, CLAIM, TURN, Engine
from ..game.tiles import N_INSTANCES, instances_of, type_of
from . import layout as L
from .world import TableWorld


@dataclass
class Signal:
    """Event timing from the table (a human's announcement)."""

    kind: str  # "turn", "after_call", "claim", "over"
    discarder: int | None = None  # for "claim": relative seat of the discarder
    events: tuple = ()  # public event kinds since the last signal, e.g. ("discard", rel_seat)


class SimTable:
    def __init__(self, world: TableWorld, seed: int, robot_seat: int, opponents, rng: np.random.Generator):
        self.w = world
        self.rng = rng
        self.engine = Engine(seed)
        self.seat = robot_seat
        self.players = {}
        it = iter(opponents)
        for s in range(4):
            if s != robot_seat:
                p = next(it)
                p.reset(s, seed * 4 + s)
                self.players[s] = p
        self.placed = np.zeros(N_INSTANCES, dtype=bool)
        self.opp_pond_count = {1: 0, 2: 0, 3: 0}
        self.opp_meld_count = {1: 0, 2: 0, 3: 0}
        self.phys_of_event: dict[int, int] = {}  # engine discard instance -> physical instance
        self.robot_pond_phys: dict[int, int] = {}  # engine discard instance -> physical (robot's discards)
        self.pending_events: list[tuple] = []
        self.shared_phys: int | None = None
        self.events_seen = 0
        self.log: list[dict] = []

    # ------------------------------------------------------------ physical tiles

    def rel(self, seat: int) -> int:
        return (seat - self.seat) % 4

    def take_physical(self, t: int) -> int:
        for i in instances_of(t):
            if not self.placed[i]:
                self.placed[i] = True
                return i
        raise RuntimeError(f"no unplaced physical tile of type {t}")

    def release_physical(self, i: int) -> None:
        self.w.deactivate(i)
        self.placed[i] = False

    def deal(self) -> None:
        """Humans deal the robot's 13 tiles upright and close-packed, in dealt order."""
        hand = list(self.engine.hands[self.seat])
        if self.engine.to_act == self.seat:  # the dealer's 14th tile is presented as a draw
            hand.remove(self.engine.last_draw)
        for k, e in enumerate(hand):
            i = self.take_physical(type_of(e))
            jitter = np.array([self.rng.normal(0, 0.0004), self.rng.normal(0, 0.0008), 0.0003])
            x = L.HAND_ANCHOR_X - (len(hand) - 1 - k) * (L.TILE_W + 0.0003)
            self.w.activate(i, np.array([x, L.HAND_Y, L.TILE_H / 2]) + jitter,
                            L.yaw_quat(self.rng.normal(0, 0.01), L.UPRIGHT_QUAT))
        self.w.settle(0.5)
        self.events_seen = len(self.engine.events)

    def present_draw(self, engine_tile: int) -> int:
        i = self.take_physical(type_of(engine_tile))
        xy = L.SHARED_XY + self.rng.uniform(-L.SHARED_JITTER_XY, L.SHARED_JITTER_XY, 2)
        yaw = self.rng.uniform(-L.SHARED_JITTER_YAW, L.SHARED_JITTER_YAW)
        self.w.activate(i, np.array([xy[0], xy[1], L.TILE_H / 2 + 0.0003]), L.yaw_quat(yaw, L.UPRIGHT_QUAT))
        self.w.settle(0.3)
        self.shared_phys = i
        return i

    def present_claimed(self, phys: int) -> None:
        """Move a claimed discard from its pond to the shared zone, face-up."""
        xy = L.SHARED_XY + self.rng.uniform(-L.SHARED_JITTER_XY, L.SHARED_JITTER_XY, 2)
        yaw = self.rng.uniform(-L.SHARED_JITTER_YAW, L.SHARED_JITTER_YAW)
        self.w.activate(phys, np.array([xy[0], xy[1], L.TILE_T / 2 + 0.0003]), L.yaw_quat(yaw))
        self.w.settle(0.3)
        self.shared_phys = phys

    def _place_opp_discard(self, seat: int, e: int) -> None:
        r = self.rel(seat)
        i = self.take_physical(type_of(e))
        k = self.opp_pond_count[r]
        self.opp_pond_count[r] += 1
        p = L.opp_pond_pos(r, k) + [self.rng.normal(0, 0.001), self.rng.normal(0, 0.001), 0.0003]
        self.w.activate(i, p, L.yaw_quat(self.rng.normal(0, 0.05)))
        self.phys_of_event[e] = i

    def _place_opp_meld_tile(self, seat: int, phys: int | None, t: int) -> None:
        r = self.rel(seat)
        i = phys if phys is not None else self.take_physical(t)
        k = self.opp_meld_count[r]
        self.opp_meld_count[r] += 1
        self.w.activate(i, L.opp_meld_pos(r, k) + [0, 0, 0.0003], L.yaw_quat(self.rng.normal(0, 0.03)))

    def _mirror_events(self) -> None:
        """Lay out the physical consequences of new engine events (the humans' actions)."""
        ev = self.engine.events
        for e in ev[self.events_seen:]:
            kind = e[0]
            if kind == "discard" and e[1] != self.seat:
                self._place_opp_discard(e[1], e[2])
                self.pending_events.append(("discard", self.rel(e[1])))
            elif kind in ("pung", "chow", "kong_exposed") and e[1] != self.seat:
                claimed, from_seat, own = e[2], e[3], e[4]
                phys = self.robot_pond_phys.pop(claimed, None) if from_seat == self.seat else self.phys_of_event.pop(claimed)
                self._place_opp_meld_tile(e[1], phys, type_of(claimed))
                for o in own:
                    self._place_opp_meld_tile(e[1], None, type_of(o))
                self.pending_events.append(("call", self.rel(e[1]), self.rel(from_seat)))
            elif kind == "kong_concealed" and e[1] != self.seat:
                for o in e[2]:
                    self._place_opp_meld_tile(e[1], None, type_of(o))
                self.pending_events.append(("kong", self.rel(e[1])))
            elif kind == "kong_added" and e[1] != self.seat:
                self._place_opp_meld_tile(e[1], None, type_of(e[2]))
                self.pending_events.append(("kong", self.rel(e[1])))
            elif kind in ("draw", "replacement_draw", "deal_draw"):
                self.pending_events.append(("draw", self.rel(e[1])))
        self.events_seen = len(ev)
        self.w.settle(0.2)

    # ------------------------------------------------------------ flow

    def advance(self) -> Signal:
        """Play other seats until the robot must decide; present its tiles; signal it."""
        eng = self.engine
        while True:
            self._mirror_events()
            if eng.done:
                return self._signal("over")
            if eng.to_act == self.seat:
                break
            s = eng.to_act
            eng.step(self.players[s].act(eng.view(s)))
        if eng.phase == TURN:
            if self.shared_phys is None:
                self.present_draw(eng.last_draw)
            return self._signal("turn")
        if eng.phase == AFTER_CALL:
            return self._signal("after_call")
        assert eng.phase == CLAIM
        return self._signal("claim", discarder=self.rel(eng.window.discard.seat))

    def _signal(self, kind: str, discarder: int | None = None) -> Signal:
        ev, self.pending_events = tuple(self.pending_events), []
        return Signal(kind, discarder, ev)

    def meld_built(self) -> None:
        """The robot has taken the claimed tile; after a Kong claim, present the replacement draw."""
        self.shared_phys = None
        eng = self.engine
        if eng.phase == TURN and eng.to_act == self.seat:
            self.present_draw(eng.last_draw)

    def legal(self) -> tuple[int, ...]:
        return self.engine.legal_actions()

    def commit(self, action: int) -> list[tuple]:
        """Apply a robot action the robot has executed and verified; return the new engine events."""
        eng = self.engine
        before = len(eng.events)
        eng.step(action)
        new = eng.events[before:]
        if action not in (A.SELF_DRAW_WIN, A.PASS, A.WIN_DISCARD):
            self.shared_phys = None  # the robot has taken the drawn or claimed tile
        for e in new:
            if e[0] in ("pung", "chow", "kong_exposed") and e[1] == self.seat:
                self.present_claimed(self.phys_of_event.pop(e[2]))
        return new

    def record_robot_discard(self, engine_tile: int, phys: int) -> None:
        self.robot_pond_phys[engine_tile] = phys

    def robot_hand_types(self) -> list[int]:
        return sorted(type_of(t) for t in self.engine.hands[self.seat])
