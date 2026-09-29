"""Play one hand with four players that act only on their own PlayerView."""

from __future__ import annotations

from dataclasses import dataclass

from .engine import Engine, IllegalAction, Result


@dataclass
class HandRecord:
    seed: int
    result: Result
    decisions: list[int]  # decision points per seat
    illegal: int  # illegal action attempts (the engine rejects them; counted, never applied)


def play_hand(seed: int, players, player_seeds: list[int] | None = None) -> HandRecord:
    """``seed`` fixes the wall; ``player_seeds`` fix each seat's own randomness."""
    engine = Engine(seed)
    player_seeds = player_seeds or [seed * 4 + s for s in range(4)]
    for s, p in enumerate(players):
        p.reset(s, player_seeds[s])
    decisions = [0, 0, 0, 0]
    illegal = 0
    while not engine.done:
        seat = engine.to_act
        action = players[seat].act(engine.view(seat))
        decisions[seat] += 1
        try:
            engine.step(action)
        except IllegalAction:
            illegal += 1
            raise
    return HandRecord(seed, engine.result, decisions, illegal)
