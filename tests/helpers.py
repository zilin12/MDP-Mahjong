"""Build walls that deal chosen tile types, for scripted rule scenarios."""

from __future__ import annotations

from mahjong_robot.game.tiles import instances_of, parse


def make_wall(hands: list[str], draws: str = "", back: str = "") -> list[int]:
    """Wall dealing ``hands`` (13 tiles each, seats 0-3), then ``draws`` from the front.

    The dealer's 14th tile is the first of ``draws``. ``back`` lists Kong replacement tiles
    in the order they are drawn from the back of the wall.
    """
    pools = {t: list(instances_of(t)) for t in range(34)}

    def take(types: list[int]) -> list[int]:
        out = []
        for t in types:
            if not pools[t]:
                raise ValueError(f"more than four copies of type {t} requested")
            out.append(pools[t].pop(0))
        return out

    front: list[int] = []
    for h in hands:
        types = parse(h)
        assert len(types) == 13, (h, len(types))
        front += take(types)
    front += take(parse(draws))
    tail = take(parse(back))
    rest = [i for t in range(34) for i in pools[t]]
    return front + rest + list(reversed(tail))
