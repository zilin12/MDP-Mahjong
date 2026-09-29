"""Gate 1 rule tests: conservation, legal transitions, calls, special wins, isolation."""

from __future__ import annotations

import copy
import random

import pytest
from helpers import make_wall

from mahjong_robot.game import actions as A
from mahjong_robot.game.engine import AFTER_CALL, CLAIM, OVER, TURN, Engine, IllegalAction
from mahjong_robot.game.hand_eval import (
    SEVEN_PAIRS,
    STANDARD,
    THIRTEEN_ORPHANS,
    is_win,
    shanten,
    winning_structure,
)
from mahjong_robot.game.tiles import counts_of, parse, type_of

E, S, RD = 27, 28, 31


def hand(text: str) -> list[int]:
    return counts_of(parse(text))


def engine(hands, draws="", back="") -> Engine:
    e = Engine(0)
    e.reset(wall=make_wall(hands, draws, back))
    return e


def pass_through(e: Engine, until_seat: int) -> None:
    """Pass every claim and discard the drawn tile until ``until_seat`` is in its turn."""
    while not (e.phase == TURN and e.to_act == until_seat):
        if e.phase == CLAIM:
            e.step(A.PASS)
        else:
            e.step(A.discard(type_of(e.last_draw)))


# ------------------------------------------------------------ winning structures


def test_standard_win_examples_from_rules():
    assert winning_structure(hand("123C 456C 789D 555B E E"), 0) == STANDARD
    assert winning_structure(hand("123C 456C 789D 555B E"), 0) is None


def test_standard_win_with_melds_counts_kong_as_one_meld():
    # Two declared melds (e.g. a Kong and a Pung) leave 2 melds + pair concealed.
    assert is_win(hand("123C 789D S S"), 2)
    assert not is_win(hand("123C 789D S S"), 1)


def test_seven_pairs_allows_four_identical_as_two_pairs():
    assert winning_structure(hand("11C 33C 66C 22D 88D 44B E E"), 0) == SEVEN_PAIRS
    assert winning_structure(hand("11C 11C 66C 22D 88D 44B E E"), 0) == SEVEN_PAIRS


def test_special_hands_require_no_melds():
    pairs = hand("11C 33C 66C 22D 88D 44B E E")
    assert not is_win([n for n in pairs], 1)
    orphans = hand("19C 19B 19D E S W N RD GD WD WD")
    assert winning_structure(orphans, 0) == THIRTEEN_ORPHANS
    assert not is_win(orphans, 1)
    assert not is_win(hand("19C 19B 19D E S W N RD GD WD 5C"), 0)


def test_shanten_agrees_with_single_tile_waits():
    rng = random.Random(3)
    wall = [t for t in range(34) for _ in range(4)]
    checked = 0
    for _ in range(3000):
        c = counts_of(rng.sample(wall, 13))
        if 4 in c:
            continue
        waits = []
        for t in range(34):
            c[t] += 1
            if is_win(c, 0):
                waits.append(t)
            c[t] -= 1
        assert (shanten(c, 0) == 0) == bool(waits), c
        checked += 1
    assert checked > 2000


def test_shanten_values():
    assert shanten(hand("123C 456C 789D 555B E E"), 0) == -1
    assert shanten(hand("123C 456C 789D 555B E"), 0) == 0
    assert shanten(hand("11C 33C 66C 22D 88D 44B E"), 0) == 0
    assert shanten(hand("19C 19B 19D E S W N RD GD WD"), 0) == 0


# ------------------------------------------------------------ transitions


SEAT0 = "123456789C 1B 9B 1D GD"
SEAT2_HONORS = "E E E S S S W W W N N N GD"


def test_deal_and_dealer_first_turn():
    e = Engine(11)
    assert [len(h) for h in e.hands] == [14, 13, 13, 13]
    assert e.phase == TURN and e.to_act == 0
    assert e.wall_remaining == 136 - 53


def test_illegal_action_rejected():
    e = Engine(5)
    missing = next(t for t in range(34) if e.counts[0][t] == 0)
    with pytest.raises(IllegalAction):
        e.step(A.discard(missing))
    with pytest.raises(IllegalAction):
        e.step(A.PUNG)


def test_heavenly_self_draw_win():
    e = engine(["123C 456C 789D 555B E", "123D 456D 789C 111B S", SEAT2_HONORS.replace("E E E", "RD RD RD"),
                "234B 567B 678D 99D RD GD"], draws="E")
    assert A.SELF_DRAW_WIN in e.legal_actions()
    e.step(A.SELF_DRAW_WIN)
    assert e.done and e.result.winner == 0 and e.result.win_type == "self_draw"


def test_nearest_single_winner_on_simultaneous_wins():
    hands = [SEAT0, "123C 456C 789C 234B RD", SEAT2_HONORS, "123D 456D 789D 567B RD"]
    e = engine(hands, draws="RD")
    e.step(A.discard(RD))
    assert e.phase == CLAIM and [s for s, _ in e.window.queue] == [1, 3]
    e.step(A.WIN_DISCARD)
    e.step(A.WIN_DISCARD)
    assert e.result.winner == 1 and e.result.from_seat == 0 and e.result.win_type == "discard"


def test_farther_winner_wins_if_nearer_passes():
    hands = [SEAT0, "123C 456C 789C 234B RD", SEAT2_HONORS, "123D 456D 789D 567B RD"]
    e = engine(hands, draws="RD")
    e.step(A.discard(RD))
    e.step(A.PASS)
    e.step(A.WIN_DISCARD)
    assert e.result.winner == 3


def test_win_beats_pung():
    hands = [SEAT0, "123C 456C 789C 234B GD", "RD RD E E E S S S W W W N N", "123D 456D 789D 567B RD"]
    e = engine(hands, draws="RD")
    e.step(A.discard(RD))
    assert [s for s, _ in e.window.queue] == [2, 3]
    e.step(A.PUNG)
    e.step(A.WIN_DISCARD)
    assert e.result.winner == 3


def test_pung_beats_chow_and_caller_must_discard():
    hands = [SEAT0, "12B 456C 789C 234D GD S", "3B 3B E E E S S W W W N N GD", "123D 456D 789D 567B RD"]
    e = engine(hands, draws="3B")
    e.step(A.discard(9 + 2))
    opts = dict(e.window.queue)
    assert A.CHOW_LOW not in opts[1] and A.CHOW_HIGH in opts[1]
    assert A.PUNG in opts[2]
    e.step(A.CHOW_HIGH)
    e.step(A.PUNG)
    assert e.phase == AFTER_CALL and e.to_act == 2
    assert e.melds[2][0].kind == "pung" and e.melds[1] == []
    assert all(a < A.CONCEALED_KONG for a in e.legal_actions())
    assert e.wall_remaining == 136 - 53  # a Pung caller does not draw


def test_chow_only_from_previous_seat():
    hands = [SEAT0, "789C 456C 123D 234D GD", "4B 5B E E E S S W W W N N GD", "123D 456D 789D 567B RD"]
    e = engine(hands, draws="3B")
    e.step(A.discard(9 + 2))
    assert e.window is None or all(s != 2 for s, _ in e.window.queue)


def test_exposed_kong_draws_replacement_from_back():
    hands = [SEAT0, "RD RD RD 456C 789C 234D S", SEAT2_HONORS, "123D 456D 789D 567B N"]
    e = engine(hands, draws="RD", back="7B")
    e.step(A.discard(RD))
    e.step(A.KONG)
    assert e.phase == TURN and e.to_act == 1
    assert e.melds[1][0].kind == "kong_exposed" and len(e.melds[1][0].tiles) == 4
    assert type_of(e.last_draw) == 9 + 6
    assert e.back == 135


def test_concealed_kong_and_replacement_win():
    hands = ["E E E 123C 456C 789C 1B", "123D 456D 789D 234B S", "RD RD RD S S W W W N N N GD GD",
             "123D 456D 789B 567B GD"]
    e = engine(hands, draws="E", back="1B")
    assert A.concealed_kong(E) in e.legal_actions()
    e.step(A.concealed_kong(E))
    assert e.melds[0][0].kind == "kong_concealed"
    assert A.SELF_DRAW_WIN in e.legal_actions()  # win on the replacement tile
    e.step(A.SELF_DRAW_WIN)
    assert e.result.structure == STANDARD


def test_added_kong_cannot_be_robbed():
    # Seat 3 holds a Thirteen Orphans hand missing only East, so it waits on East.
    hands = [SEAT0, "E E 456C 789C 234D S 9B", "RD RD RD 123D 5D 5D W W N N GD", "19C 19B 19D S W N RD GD WD WD"]
    e = engine(hands, draws="E 1C 2C 3C E", back="4C")
    e.step(A.discard(E))
    assert dict(e.window.queue)[3] == (A.PASS, A.WIN_DISCARD)
    e.step(A.PUNG)
    e.step(A.PASS)
    e.step(A.discard(9 + 8))
    pass_through(e, until_seat=1)
    assert type_of(e.last_draw) == E
    assert A.added_kong(E) in e.legal_actions()
    e.step(A.added_kong(E))
    assert e.phase == TURN and e.to_act == 1  # no claim window for seat 3, which waits on E
    assert e.melds[1][0].kind == "kong_added" and len(e.melds[1][0].tiles) == 4


def test_no_kong_when_wall_empty():
    hands = ["E E E 123C 456C 789C 1B", "123D 456D 789D 234B S", "RD RD RD S S W W W N N N GD GD",
             "123D 456D 789B 567B GD"]
    e = engine(hands, draws="E")
    e.front = e.back
    assert A.concealed_kong(E) not in e.legal_actions()


def test_wall_exhaustion_ends_hand():
    rng = random.Random(1)
    exhausted = 0
    for seed in range(200):
        e = Engine(seed)
        while not e.done:
            legal = [a for a in e.legal_actions() if a not in (A.SELF_DRAW_WIN, A.WIN_DISCARD)] or e.legal_actions()
            e.step(rng.choice(legal))
        if e.result.winner is None:
            exhausted += 1
            assert e.wall_remaining == 0 and e.events[-1] == ("exhausted",)
    assert exhausted > 150


# ------------------------------------------------------------ conservation and isolation


def test_tile_conservation_under_random_play():
    rng = random.Random(7)
    for seed in range(300):
        e = Engine(seed)
        while not e.done:
            e.step(rng.choice(e.legal_actions()))
            assert sorted(e.tile_census()) == list(range(136))
            for s in range(4):
                effective = len(e.hands[s]) + 3 * len(e.melds[s])
                assert effective in (13, 14)
        assert e.phase == OVER


def test_player_view_is_independent_of_hidden_tiles():
    rng = random.Random(2)
    for seed in range(50):
        e = Engine(seed)
        for _ in range(rng.randrange(0, 60)):
            if e.done:
                break
            e.step(rng.choice(e.legal_actions()))
        if e.done or e.wall_remaining == 0:
            continue
        before = e.view(0), e.public_events(0)
        hidden = copy.deepcopy(e)
        # Swap a tile between seat 1's concealed hand and the wall, and reshuffle the wall.
        a = hidden.hands[1][0]
        j = hidden.front
        b = hidden.wall[j]
        hidden.hands[1][0], hidden.wall[j] = b, a
        hidden.counts[1][type_of(a)] -= 1
        hidden.counts[1][type_of(b)] += 1
        rest = hidden.wall[hidden.front : hidden.back]
        rng.shuffle(rest)
        hidden.wall[hidden.front : hidden.back] = rest
        if hidden.phase == CLAIM:
            continue  # claim options of seat 1 legitimately depend on its hand
        assert (hidden.view(0), hidden.public_events(0)) == before


def test_view_exposes_only_own_concealed_tiles():
    e = Engine(4)
    v = e.view(2)
    assert set(v.hand) == set(e.hands[2])
    assert v.legal == ()  # not seat 2's decision
    assert not any(t in v.hand for s in (0, 1, 3) for t in e.hands[s])
