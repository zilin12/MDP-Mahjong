"""Decision-time search over sampled hidden states (determinized Monte Carlo rollouts).

This is the explicit POMDP layer on top of the learned policy. At a decision the policy is
not confident about, the searcher:

1. forms the belief over hidden tiles implied by the player view: every tile type's unseen
   copies (4 minus those in the own hand, all melds and all unclaimed discards), dealt
   uniformly into the opponents' concealed hands (whose sizes are public) and the live wall;
2. samples ``samples`` such hidden states, shared by all candidates (common random numbers);
3. for each of the policy's top candidate actions, applies it in every sample and plays the
   hand out with heuristic players in all four seats;
4. picks the candidate with the highest mean score (win = 1, dealing in = -``dealin_weight``),
   keeping the policy's order on ties.

It reads only the PlayerView, so hidden-information isolation holds by construction.
"""

from __future__ import annotations

import random

import numpy as np
import torch

from ..game import actions as A
from ..game.engine import AFTER_CALL, CLAIM, TURN, DiscardRecord, Engine, Meld, PlayerView, _ClaimWindow, claim_options
from ..game.tiles import N_TYPES, type_of
from .baselines import HeuristicPlayer, Player
from .features import encode
from .policy import PolicyNet, load_policy


def unseen_types(view: PlayerView) -> list[int]:
    seen = [0] * N_TYPES
    for t in view.hand:
        seen[type_of(t)] += 1
    for melds in view.melds:
        for m in melds:
            for t in m.tiles:
                seen[type_of(t)] += 1
    for pond in view.discards:
        for d in pond:
            if d.claimed_by is None:
                seen[type_of(d.tile)] += 1
    out = []
    for t in range(N_TYPES):
        out += [t] * max(0, 4 - seen[t])
    return out


def determinize(view: PlayerView, rng: random.Random) -> Engine:
    """An engine state consistent with ``view``, hidden tiles dealt uniformly at random."""
    me = view.seat
    e = Engine.__new__(Engine)
    e.rng = random.Random(rng.random())
    next_id = [t * 4 for t in range(N_TYPES)]

    def inst(t: int) -> int:
        i = next_id[t]
        next_id[t] += 1
        return i

    # Instance ids are re-issued per type (exactly four per type in total). A claimed pond
    # record's tile lives in the claimer's meld, so the record reuses a placeholder id.
    e.melds = [[Meld(m.kind, tuple(inst(type_of(t)) for t in m.tiles), None, m.from_seat) for m in view.melds[s]]
               for s in range(4)]
    e.discards = [[DiscardRecord(d.seat, inst(type_of(d.tile)) if d.claimed_by is None else type_of(d.tile) * 4,
                                 d.claimed_by) for d in view.discards[s]] for s in range(4)]
    hidden = unseen_types(view)
    rng.shuffle(hidden)
    e.hands = [[] for _ in range(4)]
    e.counts = [[0] * N_TYPES for _ in range(4)]
    for t in view.hand:
        e.hands[me].append(inst(type_of(t)))
        e.counts[me][type_of(t)] += 1
    k = 0
    for s in range(4):
        if s == me:
            continue
        for _ in range(view.concealed_counts[s]):
            t = hidden[k]
            k += 1
            e.hands[s].append(inst(t))
            e.counts[s][t] += 1
    e.wall = [inst(t) for t in hidden[k:]]
    e.front, e.back = 0, len(e.wall)
    e.events, e.result, e.last_draw = [], None, None
    if view.last_draw is not None:
        tt = type_of(view.last_draw)
        e.last_draw = next(i for i in e.hands[me] if type_of(i) == tt)
    e.phase, e.to_act, e.window = view.phase, view.to_act, None
    if view.phase == CLAIM:
        d = view.current_discard
        rec = e.discards[d.seat][-1]  # the discard under claim is the last record of its pond
        # Seats before this one in claim order are taken to have passed; later ones may claim.
        order = [(d.seat + j) % 4 for j in range(1, 4)]
        queue = []
        for s in order[order.index(me):]:
            opts = view.legal if s == me else claim_options(e.counts[s], len(e.melds[s]), type_of(rec.tile),
                                                            s == (d.seat + 1) % 4, e.wall_remaining)
            if s == me or len(opts) > 1:
                queue.append((s, tuple(opts)))
        e.window = _ClaimWindow(rec, queue)
        e.to_act = me
    return e


def _copy(e: Engine) -> Engine:
    c = Engine.__new__(Engine)
    c.rng = random.Random(0)
    c.wall, c.front, c.back = e.wall, e.front, e.back  # the wall list is never mutated; draws move indices
    c.hands = [list(h) for h in e.hands]
    c.counts = [list(x) for x in e.counts]
    c.melds = [list(m) for m in e.melds]
    c.discards = [list(d) for d in e.discards]
    c.events, c.result, c.last_draw = [], None, e.last_draw
    c.phase, c.to_act = e.phase, e.to_act
    c.window = None if e.window is None else _ClaimWindow(e.window.discard, list(e.window.queue), dict(e.window.claims))
    return c


def rollout(e: Engine, players: list[Player], me: int, dealin_weight: float) -> float:
    while not e.done:
        s = e.to_act
        e.step(players[s].act(e.view(s)))
    r = e.result
    if r.winner == me:
        return 1.0
    if r.win_type == "discard" and r.from_seat == me:
        return -dealin_weight
    return 0.0


class SearchPlayer(Player):
    name = "search"

    def __init__(self, net: PolicyNet | str, samples: int = 24, top_k: int = 3, confident: float = 0.9,
                 dealin_weight: float = 0.0):
        self.net = load_policy(net) if isinstance(net, str) else net
        self.samples, self.top_k, self.confident, self.dealin_weight = samples, top_k, confident, dealin_weight
        self.searched = 0
        self.decisions = 0

    def reset(self, seat: int, seed: int) -> None:
        super().reset(seat, seed)
        self.h = self.net.initial_state(1)
        self.search_rng = random.Random(seed * 7919 + 13)
        self.rollout_players = [HeuristicPlayer() for _ in range(4)]

    def act(self, view: PlayerView) -> int:
        obs, mask = encode(view)
        with torch.no_grad():
            logits, _, self.h = self.net.step(obs[None], mask[None], self.h)
            probs = torch.softmax(logits[0], -1).numpy()
        self.decisions += 1
        order = [int(a) for a in np.argsort(-probs) if mask[a]]
        best = order[0]
        if A.SELF_DRAW_WIN in order or A.WIN_DISCARD in order:
            return A.SELF_DRAW_WIN if A.SELF_DRAW_WIN in order else A.WIN_DISCARD  # always take a win
        if probs[best] >= self.confident or len(order) == 1 or view.phase not in (TURN, CLAIM, AFTER_CALL):
            return best
        cands = order[: self.top_k]
        self.searched += 1
        states = [determinize(view, self.search_rng) for _ in range(self.samples)]
        seeds = [self.search_rng.randrange(1 << 30) for _ in range(self.samples)]
        scores = []
        for a in cands:
            total = 0.0
            for st, sd in zip(states, seeds):
                e = _copy(st)
                for s, p in enumerate(self.rollout_players):
                    p.reset(s, sd + s)
                e.step(a)
                total += rollout(e, self.rollout_players, view.seat, self.dealin_weight)
            scores.append(total / self.samples)
        top = max(scores)
        return next(a for a, sc in zip(cands, scores) if sc == top)
