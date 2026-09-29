"""The robot seat: perception -> belief -> playing agent -> binding -> tile operations.

The robot never reads simulator ground truth. Its belief of its own hand and of the public
tiles comes from the overhead D435i and wrist D405 frames. It builds a ``PlayerView`` from the
belief, asks the playing agent for a semantic action (by tile type), binds that action to
visible tile instances, executes the tile operations, verifies the outcome with fresh
perception, and only then reports the action to the table (ADR 0002). Uncertain identity,
collision risk, or a verification that still fails after reconciliation raises ``Fault``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..agents.features import encode
from ..agents.policy import PolicyNet
from ..game import actions as A
from ..game.engine import (
    AFTER_CALL,
    CLAIM,
    TURN,
    DiscardRecord,
    Meld,
    PlayerView,
    claim_options,
    turn_actions,
)
from ..game.tiles import N_TYPES
from . import layout as L
from .cameras import D405, D435I, Frame, SimCamera
from .perception import Detection, crop_flat, crop_upright_face, detect_tiles
from .robot import CollisionRisk, look_at_rotation
from .skills import Operations, TileEstimate

CONF_MIN = 0.80  # below this, identity is uncertain
GAP_TOL = 0.0035  # a hand row with gaps no larger than this counts as close-packed
# Pushes aim past contact: the arm's absolute position carries calibration error, and a small
# overshoot only slides the free tiles beyond.
PUSH_OVERSHOOT = 0.0025


class Fault(RuntimeError):
    def __init__(self, category: str, detail: str):
        super().__init__(f"{category}: {detail}")
        self.category = category
        self.detail = detail


@dataclass
class HandTile:
    x: float
    tile_type: int
    y: float = L.HAND_Y
    yaw: float = 0.0


@dataclass
class Belief:
    seat: int
    hand: list[HandTile] = field(default_factory=list)  # sorted by x
    drawn: int | None = None  # type of the upright tile waiting in the shared zone
    own_melds: list[Meld] = field(default_factory=list)
    opp_melds: dict[int, list[int]] = field(default_factory=lambda: {1: [], 2: [], 3: []})  # rel -> types
    opp_meld_count: dict[int, int] = field(default_factory=lambda: {1: 0, 2: 0, 3: 0})
    ponds: dict[int, list[int]] = field(default_factory=lambda: {0: [], 1: [], 2: [], 3: []})  # rel -> types
    pond_dets: dict[int, list[np.ndarray]] = field(default_factory=lambda: {0: [], 1: [], 2: [], 3: []})
    current_discard: tuple[int, int] | None = None  # (rel seat, type)
    wall_remaining: int = 136 - 53
    pond_slot: int = 0

    def counts_list(self) -> list[int]:
        """Types of the believed concealed hand, including a drawn tile not yet placed."""
        return [h.tile_type for h in self.hand] + ([self.drawn] if self.drawn is not None else [])

    def counts(self) -> list[int]:
        c = [0] * N_TYPES
        for h in self.hand:
            c[h.tile_type] += 1
        if self.drawn is not None:
            c[self.drawn] += 1
        return c


class RobotSystem:  # noqa: D101 - see module docstring
    def __init__(self, world, policy: PolicyNet, classifier, contact, rng: np.random.Generator, seat: int,
                 overhead_calib=(0.001, 0.1), wrist_calib=(0.0005, 0.3)):
        self.w = world
        self.arm = world.arm
        self.ops = Operations(self.arm, contact)
        self.policy = policy
        self.h = policy.initial_state(1)
        self.clf = classifier
        self.overhead = SimCamera(world.m, D435I, rng, *overhead_calib)
        self.wrist = SimCamera(world.m, D405, rng, *wrist_calib)
        self.b = Belief(seat)
        self.log: list[dict] = []
        self.stats = {"recoveries": 0, "observations": 0, "operations": 0}
        self._scan_q: dict[str, np.ndarray] = {}
        self.debug_monitor = None  # development aid only: a ground-truth disturbance reporter
        self.public_memory: list[tuple[np.ndarray, int, float]] = []
        self.ops.locate_staged = self._locate_staged

    def close(self) -> None:
        self.overhead.close()
        self.wrist.close()

    # ------------------------------------------------------------ perception

    def observe(self) -> list[Detection]:
        """Park out of view, capture the overhead frame, detect and classify face-up tiles."""
        self.ops.park()
        self.stats["observations"] += 1
        fr = self.overhead.capture(self.w.d)
        self.last_frame = fr
        dets = detect_tiles(fr)
        bias = self.ops.contact.overhead_bias() if hasattr(self.ops.contact, "overhead_bias") else None
        if bias is not None:
            for d in dets:
                d.pos[:2] += bias
        self.last_dets = dets
        flat = [d for d in dets if not d.upright]
        crops, keep = [], []
        for d in flat:
            c = crop_flat(fr, d)
            if c is not None:
                crops.append(c)
                keep.append(d)
        types, conf = self.clf.predict(crops)
        for d, t, c in zip(keep, types, conf):
            d.tile_type, d.confidence = int(t), float(c)
        for d in dets:
            d.region = region_of(d)
        return dets

    def _locate_staged(self) -> TileEstimate | None:
        """Overhead estimate of the tile just stood at the staging spot."""
        dets = self.observe()
        near = [d for d in dets if d.upright and np.linalg.norm(d.pos[:2] - L.STAGING_XY) < 0.02]
        if len(near) != 1:
            raise Fault("verification", f"expected one tile at the staging spot, saw {len(near)}")
        return TileEstimate(near[0].pos, near[0].yaw, True)

    def _wrist_view(self, key: str, target: np.ndarray, offsets) -> Frame:
        """Move the wrist camera to look at ``target`` from the robot side; returns a frame."""
        from .skills import SAFE_Z

        if key not in self._scan_q:
            q = None
            d = self.w.d
            cid = self.wrist.cam_id
            gb = self.arm.gripper_body
            Rg = d.xmat[gb].reshape(3, 3)
            pg = d.xpos[gb]
            R_gc = Rg.T @ d.cam_xmat[cid].reshape(3, 3)
            p_gc = Rg.T @ (d.cam_xpos[cid] - pg)
            tcp_g = Rg.T @ (d.site_xpos[self.arm.tcp] - pg)
            for off, roll in offsets:
                cam = target + np.asarray(off)
                Rcam = look_at_rotation(cam, target, np.deg2rad(roll))
                Rgrip = Rcam @ R_gc.T
                tcp = cam + Rgrip @ (tcp_g - p_gc)
                if tcp[2] < 0.035:  # keep the fingertips clear of the table and tiles
                    continue
                q = self.arm.solve(tcp, Rgrip, symmetric=False)
                if q is not None:
                    break
            if q is None:
                raise Fault("planning", f"no wrist view of {key}")
            self._scan_q[key] = q
        p, R = self.arm.tcp_pose()
        if p[2] < SAFE_Z - 0.005:
            self.ops.transfer(np.array([p[0], p[1], SAFE_Z]), R)
        self.arm.free_space = True
        try:
            self.arm.move_joints(self._scan_q[key], settle=0.3)
        finally:
            self.arm.free_space = False
        return self.wrist.capture(self.w.d)

    # Camera offsets (from the target) and image rolls, most frontal first; found by an IK search
    # that keeps the fingertips at least 35 mm above the table.
    SCAN_OFFSETS = [((0.0, -0.15, 0.09), 180), ((0.0, -0.09, 0.09), 90), ((0.0, -0.18, 0.09), 90),
                    ((0.0, -0.18, 0.09), 270), ((-0.03, -0.12, 0.09), 180), ((-0.03, -0.15, 0.09), 180),
                    ((0.03, -0.15, 0.09), 180), ((0.03, -0.09, 0.12), 180)]

    def scan_hand(self, row: list[Detection]) -> list[HandTile]:
        """Read every hand tile's face with the wrist camera from two poses (left and right halves).

        Each tile is classified in every view that contains it; the most confident reading is
        kept. Positions come from the overhead row.
        """
        views = [self._wrist_view(k, np.array([x, L.HAND_Y, 0.013]), self.SCAN_OFFSETS)
                 for k, x in (("hand_left", -0.10), ("hand_right", 0.04))]
        best_t = [None] * len(row)
        best_c = np.zeros(len(row))
        for fr in views:
            crops, idx = [], []
            for i, d in enumerate(row):
                c = crop_upright_face(fr, d.pos, d.yaw)
                if c is not None:
                    crops.append(c)
                    idx.append(i)
            types, conf = self.clf.predict(crops)
            for i, t, c in zip(idx, types, conf):
                if c > best_c[i]:
                    best_t[i], best_c[i] = int(t), float(c)
        low = [i for i, c in enumerate(best_c) if c < CONF_MIN]
        if low:
            raise Fault("identity", f"uncertain hand tiles {[(round(float(row[i].pos[0]), 3), round(float(best_c[i]), 2)) for i in low]}")
        return [HandTile(d.pos[0], t, d.pos[1], d.yaw) for d, t in zip(row, best_t)]

    def read_shared_upright(self, det: Detection) -> int:
        tgt = np.array([det.pos[0], det.pos[1], 0.013])
        fr = self._wrist_view("shared", tgt, [((dx, -dy, dz), roll) for dz in (0.06, 0.08, 0.10) for dy in (0.08, 0.11)
                                              for dx in (0.0, -0.04, 0.04) for roll in (0, 90, 180, 270)])
        c = crop_upright_face(fr, det.pos, det.yaw)
        if c is None:
            raise Fault("identity", "shared-zone tile outside the wrist view")
        t, conf = self.clf.predict([c])
        if conf[0] < CONF_MIN:
            raise Fault("identity", f"uncertain drawn tile (confidence {conf[0]:.2f})")
        return int(t[0])

    # ------------------------------------------------------------ belief updates from perception

    def refresh_public(self, dets: list[Detection], discarder: int | None = None) -> None:
        """Rebuild pond and meld beliefs from a fresh observation; find the newest discard."""
        b = self.b
        for d in dets:
            if not d.region.startswith(("pond", "oppmeld", "meld")):
                continue
            # Tiles identified confidently before keep their identity (tracked by position).
            known = min(self.public_memory, key=lambda m: np.linalg.norm(m[0][:2] - d.pos[:2]), default=None)
            if known is not None and np.linalg.norm(known[0][:2] - d.pos[:2]) < 0.005 and known[2] >= d.confidence:
                d.tile_type, d.confidence = known[1], known[2]
            if d.confidence < CONF_MIN:
                raise Fault("identity", f"uncertain public tile in {d.region} (confidence {d.confidence:.2f})")
        self.public_memory = [(d.pos.copy(), d.tile_type, d.confidence) for d in dets
                              if d.region.startswith(("pond", "oppmeld", "meld"))]
        new_cur = None
        for r in range(4):
            here = [d for d in dets if d.region == f"pond{r}"]
            if r == discarder:
                prev = b.pond_dets[r]
                fresh = [d for d in here if all(np.linalg.norm(d.pos[:2] - p[:2]) > 0.008 for p in prev)]
                if not fresh:
                    raise Fault("verification", f"no new discard in pond {r}")
                # Humans fill ponds row by row, left to right: the newest is the last in that order.
                newest = max(fresh, key=lambda d: (round((d.pos[1] - L.OPP_POND[r][1]) / L.OPP_DY), d.pos[0]))
                new_cur = (r, newest.tile_type)
            b.ponds[r] = [d.tile_type for d in here]
            b.pond_dets[r] = [d.pos.copy() for d in here]
        for r in (1, 2, 3):
            b.opp_melds[r] = [d.tile_type for d in dets if d.region == f"oppmeld{r}"]
        b.current_discard = new_cur

    def won_by_discard(self) -> None:
        """The claimed winning tile joins the concealed hand (for the final consistency check)."""
        self.b.drawn = self.b.current_discard[1]

    def apply_events(self, events) -> None:
        for e in events:
            if e[0] == "draw":
                self.b.wall_remaining -= 1
            elif e[0] in ("call", "kong_concealed") and e[1] != 0:
                self.b.opp_meld_count[e[1]] += 1

    def hand_row(self, dets: list[Detection]) -> list[Detection]:
        return sorted((d for d in dets if d.region == "hand"), key=lambda d: d.pos[0])

    def sync_hand_positions(self, dets: list[Detection], expect: int) -> None:
        """Match the observed row to the believed hand by order (operations preserve order)."""
        row = self.hand_row(dets)
        if len(row) != expect or len(self.b.hand) != expect:
            raise Fault("verification", f"hand row shows {len(row)} tiles, belief {len(self.b.hand)}, expected {expect}")
        for h, d in zip(self.b.hand, row):
            h.x, h.y, h.yaw = d.pos[0], d.pos[1], d.yaw

    # ------------------------------------------------------------ playing agent

    def view(self, phase: str) -> PlayerView:
        b = self.b
        seat = b.seat
        fake = iter(range(10_000, 20_000))

        def inst(t: int) -> int:
            return t * 4 + (next(fake) % 4)

        hand = tuple(sorted(inst(h.tile_type) for h in b.hand) + ([inst(b.drawn)] if b.drawn is not None else []))
        melds = [[] for _ in range(4)]
        melds[seat] = list(b.own_melds)
        for r in (1, 2, 3):
            s = (seat + r) % 4
            tiles = tuple(inst(t) for t in b.opp_melds[r])
            n = b.opp_meld_count[r]
            if n:
                melds[s] = [Meld("pung", tiles)] + [Meld("pung", ()) for _ in range(n - 1)]
        discards = [[] for _ in range(4)]
        for r in range(4):
            s = (seat + r) % 4
            discards[s] = [DiscardRecord(s, inst(t)) for t in b.ponds[r]]
        cur = None
        if phase == CLAIM and b.current_discard is not None:
            r, t = b.current_discard
            cur = DiscardRecord((seat + r) % 4, inst(t))
        counts = b.counts()
        if phase == CLAIM:
            r, t = b.current_discard
            legal = claim_options(counts, len(b.own_melds), t, r == 3, b.wall_remaining)
        else:
            legal = turn_actions(counts, b.own_melds, b.wall_remaining, phase)
        return PlayerView(seat=seat, phase=phase, to_act=seat, legal=legal, hand=hand,
                          last_draw=inst(b.drawn) if (phase == TURN and b.drawn is not None) else None,
                          melds=tuple(tuple(m) for m in melds), discards=tuple(tuple(d) for d in discards),
                          current_discard=cur, wall_remaining=b.wall_remaining,
                          concealed_counts=(0, 0, 0, 0), result=None)

    def decide(self, phase: str) -> int:
        v = self.view(phase)
        obs, mask = encode(v)
        logits, _, self.h = self.policy.step(obs[None], mask[None], self.h)
        return int(logits[0].argmax())

    # ------------------------------------------------------------ operations with verification

    def _op(self, fn, *args):
        self.stats["operations"] += 1
        before = self.debug_monitor() if self.debug_monitor else None
        try:
            return fn(*args)
        except CollisionRisk as e:
            raise Fault("collision", str(e)) from e
        finally:
            if self.debug_monitor:
                self.debug_monitor(before, fn.__name__)

    def _extract(self, j: int) -> HandTile:
        h = self.b.hand.pop(j)
        self._op(self.ops.pick, TileEstimate(np.array([h.x, h.y, L.TILE_H / 2]), h.yaw, True, h.tile_type))
        if self.arm.gap < L.TILE_T - 0.004:
            raise Fault("verification", "extraction closed on nothing")
        return h

    def _next_pond_slot(self) -> np.ndarray:
        k = self.b.pond_slot
        if k >= L.POND_COLS * len(L.POND_ROWS_Y):
            raise Fault("planning", "robot pond is full")
        self.b.pond_slot += 1
        return L.pond_pos(k)

    def _verify_face_up(self, slot: np.ndarray, t: int, dets: list[Detection], what: str) -> Detection:
        best = min((d for d in dets if not d.upright), key=lambda d: np.linalg.norm(d.pos[:2] - slot[:2]), default=None)
        if best is None or np.linalg.norm(best.pos[:2] - slot[:2]) > 0.012:
            raise Fault("verification", f"{what}: no face-up tile at its slot")
        if best.confidence < CONF_MIN or best.tile_type != t:
            raise Fault("verification", f"{what}: slot shows type {best.tile_type} ({best.confidence:.2f}), expected {t}")
        return best

    def _observe_verified(self, check) -> list[Detection]:
        """Observe and run ``check``; on failure observe once more (reconciliation) before faulting."""
        dets = self.observe()
        try:
            check(dets)
            return dets
        except Fault:
            self.stats["recoveries"] += 1
            dets = self.observe()
            check(dets)
            return dets

    def _find_shared(self, dets, upright: bool) -> Detection:
        cands = [d for d in dets if d.region == "shared" and d.upright == upright]
        if len(cands) != 1:
            raise Fault("verification", f"expected one {'upright' if upright else 'face-up'} tile in the shared zone, saw {len(cands)}")
        return cands[0]

    def _pick_shared(self, dets, upright: bool) -> None:
        d = self._find_shared(dets, upright)
        self._op(self.ops.pick, TileEstimate(d.pos, d.yaw, upright, d.tile_type))
        width = L.TILE_T if upright else L.TILE_W
        if self.arm.gap < width - 0.004:
            raise Fault("verification", "shared-zone pick closed on nothing")

    def compact(self, expect: int) -> None:
        """Close gaps in the hand row (ADR 0004).

        Repeatedly take the outermost gap on each side and push the smaller outer segment
        (between that gap and its row end) toward it, so the pusher always enters from free
        space beside the row.
        """
        pusher_half = 0.0075
        for _ in range(10):
            dets = self.observe()
            row = self.hand_row(dets)
            xs = [d.pos[0] for d in row]
            gaps = [(i, xs[i + 1] - xs[i] - L.TILE_W) for i in range(len(xs) - 1)
                    if xs[i + 1] - xs[i] - L.TILE_W > GAP_TOL]
            if not gaps:
                self.sync_hand_positions(dets, expect)
                return
            (i_l, g_l), (i_r, g_r) = gaps[0], gaps[-1]
            left_size, right_size = i_l + 1, len(xs) - 1 - i_r
            if left_size <= right_size:  # push the left end segment rightward
                edge = xs[0] - L.TILE_W / 2
                self._op(self.ops.push, edge - pusher_half - 0.004, edge + g_l + PUSH_OVERSHOOT - pusher_half, L.HAND_Y)
            else:  # push the right end segment leftward
                edge = xs[-1] + L.TILE_W / 2
                self._op(self.ops.push, edge + pusher_half + 0.004, edge - g_r - PUSH_OVERSHOOT + pusher_half, L.HAND_Y)
        raise Fault("verification", "hand row still has gaps after compaction")

    def _insert_drawn(self) -> None:
        """Take the drawn tile from the shared zone and set it against the roomier free end."""
        self._pick_shared(self.observe(), upright=True)
        if self.b.hand:
            lo, hi = min(h.x for h in self.b.hand), max(h.x for h in self.b.hand)
            if (lo - L.HAND_X_MIN) >= (L.HAND_X_MAX - hi):
                x, at, side = lo - L.TILE_W - L.INSERT_MARGIN, 0, "left"
            else:
                x, at, side = hi + L.TILE_W + L.INSERT_MARGIN, len(self.b.hand), "right"
            if not L.HAND_X_MIN - 0.005 <= x <= L.HAND_X_MAX + 0.005:
                raise Fault("planning", f"no room to insert at x={x:.3f}")
        else:
            x, at, side = L.HAND_ANCHOR_X, 0, "left"
        self._op(self.ops.insert, np.array([x, L.HAND_Y, 0.0]), 0.0, side)
        self.b.hand.insert(at, HandTile(x, self.b.drawn))
        self.b.drawn = None

    # ------------------------------------------------------------ turns

    def start(self) -> None:
        """Identify the dealt hand (overhead positions, wrist faces)."""
        dets = self.observe()
        row = self.hand_row(dets)
        self.b.hand = self.scan_hand(row)
        self.refresh_public(dets)

    def on_turn(self) -> int:
        """Robot's turn with a drawn tile standing in the shared zone. Returns the committed action."""
        dets = self._observe_public()
        shared = self._find_shared(dets, upright=True)
        self.b.drawn = self.read_shared_upright(shared)
        action = self.decide(TURN)
        if action == A.SELF_DRAW_WIN:
            return action
        n = len(self.b.hand)
        if action < A.CONCEALED_KONG:
            self._discard(action - A.DISCARD, dets, n)
        elif action < A.ADDED_KONG:
            self._concealed_kong(action - A.CONCEALED_KONG, dets)
        else:
            self._added_kong(action - A.ADDED_KONG, dets)
        return action

    def _discard(self, t: int, dets, n: int, from_turn: bool = True) -> None:
        drawn = self.b.drawn
        slot = self._next_pond_slot()
        if from_turn and drawn == t:
            self._pick_shared(dets, upright=True)
            self._op(self.ops.tip_over, slot)
            self.b.drawn = None
        else:
            j = next((k for k, h in enumerate(self.b.hand) if h.tile_type == t), None)
            if j is None:
                raise Fault("belief", f"no {t} in the believed hand")
            self._extract(j)
            self._op(self.ops.tip_over, slot)
            if from_turn:
                self._insert_drawn()
        self._observe_verified(lambda ds: self._verify_face_up(slot, t, ds, "discard"))
        self.compact(len(self.b.hand))

    def on_after_call(self) -> int:
        """After a Chow/Pung the robot must discard from its hand."""
        action = self.decide(AFTER_CALL)
        if not action < A.CONCEALED_KONG:
            raise Fault("belief", "after a call only a discard is legal")
        self._discard(action - A.DISCARD, None, len(self.b.hand), from_turn=False)
        return action

    def _meld_slot(self, meld_index: int, tile_index: int) -> np.ndarray:
        if meld_index >= 4:
            raise Fault("planning", "no meld slot left")
        return L.meld_pos(meld_index, tile_index)

    def _concealed_kong(self, t: int, dets) -> None:
        m = len(self.b.own_melds)
        k = 0
        if self.b.drawn == t:
            self._pick_shared(dets, upright=True)
            self._op(self.ops.tip_over, self._meld_slot(m, k))
            self.b.drawn = None
            k += 1
        while k < 4:
            j = next(i for i, h in enumerate(self.b.hand) if h.tile_type == t)
            self._extract(j)
            self._op(self.ops.tip_over, self._meld_slot(m, k))
            k += 1
        if self.b.drawn is not None:  # the drawn tile joins the hand
            self._insert_drawn()
        self.b.own_melds.append(Meld("kong_concealed", tuple(t * 4 + i for i in range(4))))
        self.compact(len(self.b.hand))
        self._observe_verified(lambda ds: [self._verify_face_up(self._meld_slot(m, i), t, ds, "kong") for i in range(4)])

    def _added_kong(self, t: int, dets) -> None:
        mi = next(i for i, mm in enumerate(self.b.own_melds) if mm.kind == "pung" and mm.tile_type == t)
        slot = self._meld_slot(mi, 3)
        if self.b.drawn == t:
            self._pick_shared(dets, upright=True)
            self._op(self.ops.tip_over, slot)
            self.b.drawn = None
        else:
            j = next(i for i, h in enumerate(self.b.hand) if h.tile_type == t)
            self._extract(j)
            self._op(self.ops.tip_over, slot)
            self._insert_drawn()
        old = self.b.own_melds[mi]
        self.b.own_melds[mi] = Meld("kong_added", old.tiles + (t * 4 + 3,))
        self._observe_verified(lambda ds: self._verify_face_up(slot, t, ds, "added kong"))
        self.compact(len(self.b.hand))

    def _observe_public(self, discarder: int | None = None) -> list[Detection]:
        """Observe and update public beliefs; one fresh observation on failure (reconciliation)."""
        dets = self.observe()
        try:
            self.refresh_public(dets, discarder)
        except Fault:
            self.stats["recoveries"] += 1
            dets = self.observe()
            self.refresh_public(dets, discarder)
        return dets

    def on_claim(self, discarder: int) -> int:
        self._observe_public(discarder)
        return self.decide(CLAIM)

    def build_meld(self, action: int, claimed_type: int) -> None:
        """After a won Chow/Pung/Kong claim: the claimed tile waits face-up in the shared zone."""
        m = len(self.b.own_melds)
        dets = self.observe()
        shared = self._find_shared(dets, upright=False)
        if shared.confidence < CONF_MIN or shared.tile_type != claimed_type:
            raise Fault("identity", f"shared zone shows {shared.tile_type}, claimed {claimed_type}")
        if action == A.PUNG:
            own, kind = [claimed_type, claimed_type], "pung"
        elif action == A.KONG:
            own, kind = [claimed_type] * 3, "kong_exposed"
        else:
            own, kind = [claimed_type + d for d in A.CHOW_OFFSETS[action]], "chow"
        self._pick_shared(dets, upright=False)
        self._op(self.ops.place_flat, self._meld_slot(m, 0), 0.0)
        for k, t in enumerate(own, start=1):
            j = next((i for i, h in enumerate(self.b.hand) if h.tile_type == t), None)
            if j is None:
                raise Fault("belief", f"no {t} in the believed hand for the meld")
            self._extract(j)
            self._op(self.ops.tip_over, self._meld_slot(m, k))
        tiles = tuple(t * 4 for t in [claimed_type] + own)
        self.b.own_melds.append(Meld(kind, tiles, claimed=tiles[0]))
        types = [claimed_type] + own
        self._observe_verified(lambda ds: [self._verify_face_up(self._meld_slot(m, i), t, ds, "meld")
                                           for i, t in enumerate(types)])
        self.compact(len(self.b.hand))


def region_of(d: Detection) -> str:
    x, y = d.pos[0], d.pos[1]
    if np.linalg.norm(np.array([x, y]) - L.SHARED_XY) < 0.03:
        return "shared"
    if d.upright and abs(y - L.HAND_Y) < 0.012 and L.HAND_X_MIN - 0.015 < x < L.HAND_X_MAX + 0.02:
        return "hand"
    if not d.upright:
        for mx, my in L.MELD_SLOTS:
            if abs(x - mx) < 0.015 and abs(y - my) < 0.015:
                return "meld0"
        if L.POND_X0 - 0.02 < x < L.POND_X0 + L.POND_COLS * L.POND_DX and 0.21 < y < 0.32:
            return "pond0"
        for r, (x0, y0) in L.OPP_POND.items():
            if x0 - 0.02 < x < x0 + L.OPP_COLS * L.OPP_DX and y0 - 0.02 < y < y0 + 5 * L.OPP_DY - 0.01:
                return f"pond{r}"
        for r, (x0, y0) in L.OPP_MELD.items():
            if x0 - 0.02 < x < x0 + L.OPP_COLS * L.OPP_DX and y0 - 0.02 < y < y0 + 4 * 0.034:
                return f"oppmeld{r}"
    return "other"
