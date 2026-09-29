"""Batched rollout collection for the robot seat.

Each slot plays one hand. Opponents act inline; robot-seat decisions from all slots are
batched through the policy. The robot seat is ``seed % 4`` so training is seat-balanced.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np
import torch

from ..agents.baselines import PLAYERS, Player
from ..agents.features import encode, potential
from ..agents.policy import PolicyNet, PolicyPlayer, load_policy
from ..game.engine import Engine

_POLICY_CACHE: dict[str, PolicyNet] = {}


def make_player(spec: str) -> Player:
    """'random', 'heuristic', 'heldout_heuristic', or 'policy:<checkpoint path>'."""
    if spec in PLAYERS:
        return PLAYERS[spec]()
    if spec.startswith("search:"):
        # search:<checkpoint>[:samples[:top_k[:confident]]]
        from ..agents.search import SearchPlayer

        parts = spec.split(":")
        path = parts[1]
        if path not in _POLICY_CACHE:
            _POLICY_CACHE[path] = load_policy(path)
        kw = {}
        for key, val in zip(("samples", "top_k", "confident"), parts[2:]):
            kw[key] = float(val) if key == "confident" else int(val)
        return SearchPlayer(_POLICY_CACHE[path], **kw)
    if spec.startswith("policy:"):
        path = spec.split(":", 1)[1]
        if path not in _POLICY_CACHE:
            _POLICY_CACHE[path] = load_policy(path)
        return PolicyPlayer(_POLICY_CACHE[path], greedy=True)
    raise ValueError(f"unknown player spec {spec!r}")


@dataclass
class Episode:
    seed: int
    seat: int
    obs: list = field(default_factory=list)
    mask: list = field(default_factory=list)
    act: list = field(default_factory=list)
    logp: list = field(default_factory=list)
    val: list = field(default_factory=list)
    pot: list = field(default_factory=list)
    rew: np.ndarray | None = None
    outcome: dict | None = None

    def pack(self) -> dict:
        return {
            "seed": self.seed,
            "seat": self.seat,
            "obs": np.stack(self.obs),
            "mask": np.stack(self.mask),
            "act": np.array(self.act, dtype=np.int64),
            "logp": np.array(self.logp, dtype=np.float32),
            "val": np.array(self.val, dtype=np.float32),
            "rew": self.rew,
            "outcome": self.outcome,
        }


class _Slot:
    def __init__(self, seed: int, opponents: list[str], shaping: float):
        self.engine = Engine(seed)
        self.seat = seed % 4
        self.players: list[Player | None] = []
        it = iter(opponents)
        for s in range(4):
            if s == self.seat:
                self.players.append(None)
            else:
                p = make_player(next(it))
                p.reset(s, seed * 4 + s)
                self.players.append(p)
        self.ep = Episode(seed, self.seat)
        self.shaping = shaping

    def advance(self) -> bool:
        """Run opponents until the robot must act. Returns False when the hand is over."""
        e = self.engine
        while not e.done and e.to_act != self.seat:
            s = e.to_act
            e.step(self.players[s].act(e.view(s)))
        return not e.done

    def finish(self, dealin_penalty: float) -> dict | None:
        ep = self.ep
        r = self.engine.result
        won = r.winner == self.seat
        dealt_in = r.win_type == "discard" and r.from_seat == self.seat
        ep.outcome = {
            "win": won,
            "dealin": dealt_in,
            "exhausted": r.winner is None,
            "win_type": r.win_type if won else None,
            "steps": len(ep.act),
        }
        if not ep.act:
            return None
        pots = ep.pot + [0.0]
        rew = self.shaping * (np.array(pots[1:]) - np.array(pots[:-1]))
        rew[-1] += (1.0 if won else 0.0) - (dealin_penalty if dealt_in else 0.0)
        ep.rew = rew.astype(np.float32)
        return ep.pack()


def collect(
    net: PolicyNet,
    seeds: list[int],
    opponent_sampler,
    shaping: float = 0.0,
    dealin_penalty: float = 0.0,
    batch: int = 32,
    greedy: bool = False,
    torch_seed: int = 0,
) -> tuple[list[dict], list[dict]]:
    """Play every hand in ``seeds``; return (packed episodes, outcomes of all hands)."""
    gen = torch.Generator().manual_seed(torch_seed)
    todo = list(seeds)
    slots: list[_Slot | None] = [None] * batch
    h = net.initial_state(batch)
    episodes, outcomes = [], []

    def refill(i: int) -> None:
        while todo:
            seed = todo.pop(0)
            slot = _Slot(seed, opponent_sampler(random.Random(seed)), shaping)
            h[:, i] = 0.0
            if slot.advance():
                slots[i] = slot
                return
            slot.finish(dealin_penalty)
            outcomes.append(slot.ep.outcome)
        slots[i] = None

    for i in range(batch):
        refill(i)
    while any(s is not None for s in slots):
        active = [i for i, s in enumerate(slots) if s is not None]
        views = [slots[i].engine.view(slots[i].seat) for i in active]
        enc = [encode(v) for v in views]
        obs = np.stack([o for o, _ in enc])
        mask = np.stack([m for _, m in enc])
        logits, values, h_new = net.step(obs, mask, h[:, active])
        h[:, active] = h_new
        if greedy:
            acts = logits.argmax(-1)
        else:
            acts = torch.multinomial(torch.softmax(logits, -1), 1, generator=gen)[:, 0]
        logps = torch.log_softmax(logits, -1).gather(1, acts[:, None])[:, 0]
        for j, i in enumerate(active):
            slot = slots[i]
            a = int(acts[j])
            ep = slot.ep
            ep.obs.append(obs[j])
            ep.mask.append(mask[j])
            ep.act.append(a)
            ep.logp.append(float(logps[j]))
            ep.val.append(float(values[j]))
            ep.pot.append(potential(views[j]))
            slot.engine.step(a)
            if not slot.advance():
                packed = slot.finish(dealin_penalty)
                outcomes.append(slot.ep.outcome)
                if packed is not None:
                    episodes.append(packed)
                refill(i)
    return episodes, outcomes
