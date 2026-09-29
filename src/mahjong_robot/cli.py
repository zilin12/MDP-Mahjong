"""Command line entry points: ``mahjong-robot <command>``."""

from __future__ import annotations

import argparse
import json


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="mahjong-robot")
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="train the playing agent (gate 1)")
    t.add_argument("--config", default="configs/gate1.yaml")
    t.add_argument("--run-dir")
    t.add_argument("--iterations", type=int)

    e = sub.add_parser("gate1", help="evaluate a policy on held-out paired hands")
    e.add_argument("--policy", required=True)
    e.add_argument("--out", default="reports/gate1")
    e.add_argument("--hands", type=int, default=10_000)
    e.add_argument("--opponents", default="heldout_heuristic")
    e.add_argument("--processes", type=int, default=20)
    e.add_argument("--test-block", type=int, default=0, help="held-out seed block (0 = original test set)")

    p = sub.add_parser("play", help="print one hand move by move")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--players", default="heuristic,heuristic,heuristic,heuristic",
                   help="four comma-separated player specs (e.g. policy:runs/gate1/checkpoints/best.pt)")

    s = sub.add_parser("sim", help="gate 3: play hands in the MuJoCo physics simulator")
    s.add_argument("--policy", required=True)
    s.add_argument("--hands", type=int, default=100)
    s.add_argument("--mode", default="hybrid", choices=["hybrid", "classical", "learned"])
    s.add_argument("--out", default="reports/gate3")
    s.add_argument("--processes", type=int, default=20)
    s.add_argument("--start", type=int, default=0, help="first held-out hand index")

    rc = sub.add_parser("record", help="film one simulated hand (MP4 plus a game record)")
    rc.add_argument("--seed", type=int, default=3000000265, help="hand seed (the robot sits in seat seed %% 4)")
    rc.add_argument("--mode", default="classical", choices=["hybrid", "classical", "learned"])
    rc.add_argument("--policy", default="models/playing_policy.pt")
    rc.add_argument("--out", default="reports/videos")
    rc.add_argument("--every", type=int, default=40, help="physics steps per video frame (40 = x2.4 at 30 fps)")

    tr = sub.add_parser("trial", help="record or score a physical trial (gates 2 and 4)")
    tr.add_argument("op", choices=["add", "report"])
    tr.add_argument("--gate", type=int, choices=[2, 4], required=True)
    tr.add_argument("--kind", help="skill, 'recognition', 'hand', or a call form / win route")
    tr.add_argument("--success", choices=["yes", "no"])
    tr.add_argument("--reason", default="")
    tr.add_argument("--tile-type", type=int)
    tr.add_argument("--corrective-touch", action="store_true")
    tr.add_argument("--note", default="")

    args = ap.parse_args(argv)
    if args.cmd == "train":
        from .rl.train import TrainConfig, train

        cfg = TrainConfig.load(args.config)
        if args.run_dir:
            cfg.run_dir = args.run_dir
        if args.iterations:
            cfg.iterations = args.iterations
        train(cfg)
    elif args.cmd == "gate1":
        from .rl.evaluate import gate1

        rep = gate1(args.policy, args.out, args.hands, args.processes, args.opponents, args.test_block)
        print(json.dumps({k: rep[k] for k in ("passed", "criteria", "policy_minus_random",
                                               "policy_minus_heuristic")}, indent=2))
        for name, c in rep["conditions"].items():
            print(f"{name:10s} win {c['win_rate']:.4f} dealin {c['dealin_rate']:.4f} illegal {c['illegal_actions']}")
    elif args.cmd == "play":
        from .game import actions as A
        from .game.engine import Engine
        from .game.tiles import name, type_of
        from .rl.collect import make_player

        players = [make_player(sp) for sp in args.players.split(",")]
        eng = Engine(args.seed)
        for i, pl in enumerate(players):
            pl.reset(i, args.seed * 4 + i)
        while not eng.done:
            seat = eng.to_act
            a = players[seat].act(eng.view(seat))
            hand = " ".join(name(type_of(x)) for x in sorted(eng.hands[seat]))
            print(f"seat {seat} [{eng.phase:10s}] {A.describe(a):24s} | {hand}")
            eng.step(a)
        print("result:", eng.result)
    elif args.cmd == "trial":
        from . import trials

        if args.op == "add":
            row = {"kind": args.kind, "success": args.success == "yes", "reason": args.reason,
                   "corrective_touch": args.corrective_touch, "note": args.note}
            if args.tile_type is not None:
                row["tile_type"] = args.tile_type
            print(json.dumps(trials.record(args.gate, row)))
        else:
            rows = trials.load(args.gate)
            rep = trials.score_gate2(rows) if args.gate == 2 else trials.score_gate4(rows)
            print(json.dumps(rep, indent=2))
    elif args.cmd == "record":
        from pathlib import Path

        from .sim.gate3 import play_hand
        from .sim.video import HandRecorder

        path = Path(args.out) / f"hand_{args.seed}_{args.mode}.mp4"
        title = f"hand {args.seed} · {args.mode} · {Path(args.policy).name}"
        rep = play_hand(args.seed, args.mode, args.policy,
                        observer=lambda w, t, r: HandRecorder(w, t, r, path, every=args.every, title=title))
        print(json.dumps({k: rep[k] for k in ("completed", "outcome", "fault", "sim_time_s", "operations")}))
        print(f"video: {path}\nrecord: {path.with_suffix('.md')}")
    elif args.cmd == "sim":
        from .sim.gate3 import run_gate3

        rep = run_gate3(args.policy, args.hands, args.mode, args.out, args.processes, args.start)
        print(json.dumps(rep["summary"], indent=2))


if __name__ == "__main__":
    main()
