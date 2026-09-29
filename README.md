# Single-arm Mahjong

One YAM arm (linear_4310 gripper, wrist RealSense D405, overhead D435i) plays one seat of a simplified Chinese Mahjong hand, from the game decision through physical tile manipulation. The plan is [PLAN.md](PLAN.md), the vocabulary is [CONTEXT.md](CONTEXT.md), the rules are [docs/rules.md](docs/rules.md) plus [ADR 0001](docs/adr/0001-rule-profile.md), and the gate criteria are [docs/evaluation.md](docs/evaluation.md).

## Status by gate

See [STATUS.md](STATUS.md) for measured results and what remains.

| Gate | Where it runs | State |
| --- | --- | --- |
| 1. Playing agent | symbolic engine | passed; accepted policy wins 29.8% against the held-out heuristic (random 0.07%), 0 illegal actions |
| 2. Tile operations | physical robot | not run on hardware; simulation counterpart meets both thresholds |
| 3. Physics integration | MuJoCo | with the accepted policy: hybrid 71/100 and classical-only 84/100 completed (both meet 70); learned-only 0/100 |
| 4. Physical play | physical robot | not run (no hardware in this environment); protocol and scoring ready |

## Layout

```text
src/mahjong_robot/
  game/      rules engine (fixed rule profile), tiles, hand evaluation, table runner
  agents/    baselines (random, heuristic, held-out heuristic), features, recurrent policy
  rl/        batched rollouts, recurrent PPO with curriculum and league, gate-1 evaluation
  sim/       MuJoCo scene (vendored i2rt YAM + linear_4310), cameras, perception,
             IK and control, tile operations, learned contact policy, the simulated table,
             the robot system, and the gate-2 and gate-3 runners
  vision/    tile-face dataset rendering and classifier
  trials.py  hardware trial logging and scoring (gates 2 and 4)
assets/i2rt  vendored YAM v1 and linear_4310 models (MIT, see assets/i2rt/SOURCE.md)
configs/     training configuration
docs/        rules, ADRs, evaluation criteria, hardware protocols
reports/     gate reports and raw trial logs
```

## Running

```bash
uv sync --extra sim
uv run pytest                      # rules, legality, trial scoring (fast)
uv run pytest -m slow              # physics and perception checks

# Gate 1
uv run mahjong-robot train --config configs/gate1.yaml
uv run mahjong-robot gate1 --policy runs/gate1/checkpoints/best.pt --hands 10000

# One hand in the symbolic engine, move by move
uv run mahjong-robot play --seed 3 --players policy:runs/gate1/checkpoints/best.pt,heuristic,heuristic,heuristic

# Gate 3 (needs models/tile_classifier.pt and models/contact_policy.pt)
MUJOCO_GL=egl uv run mahjong-robot sim --policy models/playing_policy.pt --mode hybrid --hands 100

# Hardware trials (gates 2 and 4)
uv run mahjong-robot trial add --gate 4 --kind hand --success yes
uv run mahjong-robot trial report --gate 4
```

The models are regenerated with `scripts/build_tile_dataset.py` plus `vision.classifier.train`, and `scripts/generate_contact_data.py` plus `sim.learned.train_models`. The exact commands are in STATUS.md.
