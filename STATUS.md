# Status

Measured on 2026-09-28. Every number below comes from a report in `reports/`. Simulation and hardware are reported separately; no hardware trial has been run (this environment has no robot).

| Gate | Criterion | Result |
| --- | --- | --- |
| 1. Playing agent | no illegal actions; paired win-rate advantage over legal random, 95% CI above 0, over 10,000 held-out, seat-balanced hands | **Passed.** Accepted policy: 29.8% win rate against the held-out heuristic (goal 30%) |
| 2. Tile operations | recognition ≥90% overall and ≥80% per type; each skill ≥90% over 50 held-out **physical** trials | **Not run on hardware.** The simulation counterpart meets both thresholds |
| 3. Physics integration | ≥70 of 100 held-out hands completed; compare hybrid with classical-only and learned-only | **With the accepted policy: hybrid 71/100 (meets 70), classical-only 84/100, learned-only 0/100** |
| 4. Physical play | ≥3/5 hands without corrective touch; ≥8/10 per call form and win route | **Not run** (needs hardware); protocol and scoring ready |

## Gate 1 — playing agent (`reports/gate1/`)

Recurrent (GRU) actor-critic with legal-action masking, trained with PPO in three stages: random opponents, then the training heuristic, then a league mixing the heuristic with saved snapshots. The run took 300 iterations of 4,096 hands, and the checkpoint was selected on validation seeds (`runs/gate1/best.json`, iteration 289).

Win rate in the robot seat over 10,000 held-out, paired hands, with the robot seat `seed % 4` (2,500 hands per seat):

| Opponents (×3) | Policy | Legal random | Training heuristic | Policy − random (95% CI) |
| --- | --- | --- | --- | --- |
| held-out heuristic (gate) | 13.4% | 0.11% | 30.3% | +13.3 pp [+12.6, +14.0] |
| training heuristic | 11.0% | 0.14% | 24.4% | +10.9 pp [+10.3, +11.5] |
| legal random | 70.2% | 0.84% | 88.5% | +69.3 pp [+68.4, +70.2] |

- There were zero illegal actions in every condition (the engine rejects them; none were attempted).
- The learned policy is clearly weaker than the fixed heuristic, by 13–18 pp. PLAN.md requires only the random comparison and asks for the heuristic comparison to be reported.
- Nearly all of the policy's wins are Standard wins; Seven Pairs occurred 2 times in 10,000 hands and Thirteen Orphans never.

## Gate 1, second policy (29 September 2026; `runs/gate1_v2/`, `reports/gate1_v2/`)

Goal: a 30% robot-seat win rate against the held-out heuristic, which is the heuristic's own level. Method:

1. **Imitation start.** The recurrent policy copies the training heuristic from 150,000 hands (300,000 seat-episodes). It matches the heuristic's choice 80.7% of the time; the rest are mostly the heuristic's random tie-breaks.
2. **PPO fine-tuning.** 150 iterations in the league stage with a KL penalty toward the imitation policy, a 0.2 deal-in penalty, a constant exploration bonus, GPU updates and fresh training seeds. Validation stayed flat at 25–27%, and the selected checkpoint is iteration 74 (26.8%, against 25.9% at the start).
3. **Decision-time search** (`agents/search.py`, determinized rollouts over sampled hidden tiles) was tried on validation hands in four configurations. None improved on the policy (−3.4 to −0.05 points; `search_tuning.json`), so it was not adopted.

Result on a **fresh held-out test block** (block 1, seeds 2,000,100,000–2,000,109,999; block 0 had already been seen), 10,000 paired, seat-balanced hands, held-out heuristic opponents:

| Robot seat | Win rate (95% CI) | Deal-in rate |
| --- | --- | --- |
| Second policy | **29.8%** [28.9, 30.7] | 18.3% |
| Training heuristic | 30.5% [29.6, 31.4] | 18.4% |
| Legal random | 0.07% | 27.7% |

- Policy minus random: +29.7 pp [28.8, 30.6]. Policy minus heuristic: −0.8 pp [−1.6, +0.1]. Zero illegal actions.
- The point estimate is 0.2 points under 30%, and the confidence interval contains 30%. It is statistically indistinguishable from the heuristic, and up from 13.4% for the first policy.
- The Gate 1 pass criterion in `docs/evaluation.md` is unchanged (beat random). Gate 3 still uses the first policy.

## Gate 2 — tile operations

**Hardware:** not run. [docs/gate2-protocol.md](docs/gate2-protocol.md) lists the checks to do before the first trial (gripper direction, pad width, tile size, calibration), and `mahjong-robot trial add/report --gate 2` records and scores the trials.

**Simulation counterpart** (this is not gate evidence):

- **Recognition** (`reports/gate2/recognition_sim.json`): 98.1% overall on 680 held-out views (10 overhead face-up and 10 wrist face-on views per type); the worst type scores 90%. Overhead views are 100% correct and wrist views 96.2%. Misreads have a mean confidence of 0.07, so the 0.8 threshold turns them into uncertain-identity faults.
- **Skills** (`reports/gate2/skill_trials_sim_summary.json`): 50 final held-out trials per skill (seeds from 60,000, disjoint from development), with targets from overhead perception:

| Skill | Classical | Hybrid | Learned-only |
| --- | --- | --- | --- |
| Shared-zone pick | 49 | 49 | 0 |
| Close-packed extraction | 48 | 48 | 0 |
| Insertion | 49 | 49 | 0 |
| Reorientation (tip-over) | 50 | 50 | 0 |
| Placement | 50 | 50 | 0 |

## Gate 3 with the accepted policy (`reports/gate3/final_v2/`)

Held-out block 2 (hands 200–299), used for the first time, with `models/playing_policy.pt`. Everything else is unchanged from the block-1 run below.

| | Hybrid | Classical-only | Learned-only |
| --- | --- | --- | --- |
| Completed | **71** | **84** | **0** |
| Robot won / other seat won / exhausted | 15 / 56 / 0 | 20 / 64 / 0 | — |
| Uncertain identity | 15 | 6 | — |
| Verification failed | 13 | 9 | — |
| Silent identity error | 1 | 1 | — |
| Collision risk | 0 | 0 | 100 |
| Simulated time per completed hand | 541 s | 552 s | — |
| Tile operations per hand | 33.5 | 35.9 | 0 |
| Recoveries (hands completed after one) | 41 (8) | 36 (15) | 0 |

- **Both modes now meet the 70-hand bar.** Paired on the same hands, classical completed 13 hands the hybrid did not, and the hybrid none that classical did not (exact McNemar p = 0.0002). The learned flat-pick refinement and online bias correction (ADR 0005) measurably hurt.
- **Why completion rose from block 1.** The accepted policy wins more (20 robot wins in 100 classical hands, against 5), and its hands need fewer tile operations (35.9 per hand, against 42.8), leaving fewer chances to fault. The seed blocks also differ, so the before-and-after comparison is not paired.
- Learned-only still fails on its first free-space move.

## Gate 3 — physics integration, first policy (`reports/gate3/final/`)

The full loop runs in MuJoCo:

- **Perception:** overhead D435i and wrist D405 images, with injected calibration error and stereo depth noise.
- **Belief:** built from perception only, with the Gate 1 policy choosing actions.
- **Execution:** binding to tile instances, the five tile operations, fresh-perception verification with one reconciliation, and faults on uncertainty.
- **Scoring:** ground truth is used only to score. Opponents are three heuristics played by the simulated table.

Final held-out block 1 (100 hands, identical seeds for every mode):

| | Hybrid | Classical-only | Learned-only |
| --- | --- | --- | --- |
| Completed | **66** | **72** | **0** |
| Robot won / other seat won / exhausted | 4 / 61 / 1 | 5 / 66 / 1 | — |
| Uncertain identity | 18 | 15 | — |
| Verification failed | 14 | 11 | — |
| Silent identity error (ground-truth check) | 2 | 2 | — |
| Collision risk | 0 | 0 | 100 |
| Simulated time per completed hand | 704 s | 706 s | — |
| Tile operations per hand | 40.6 | 42.8 | 0 |
| Recoveries (hands completed after one) | 50 (11) | 51 (11) | 0 |

- **Paired comparison:** 65 hands were completed by both; 7 by classical only and 1 by hybrid only (exact McNemar p = 0.07).
- **Preliminary run:** an earlier classical-only run on held-out block 0 completed 75/100 (`reports/gate3/preliminary_block0/`). It came before the last changes (the learned-policy fixes and fault labelling), so block 1 was used for the final comparison.

**What this shows.** The perception-to-execution loop completes about 70% of simulated hands with classical contact control. The learned wrist-camera contact policy did not beat open-loop classical contact in this simulator ([ADR 0005](docs/adr/0005-hybrid-scope.md)). Development trials showed it hurting close-packed extraction and right-end insertion, so the hybrid uses it only for flat picks; even there, it plus the online overhead-bias correction completed 6 fewer hands than classical. The learned-only ablation fails on its first free-space move. Its goal-conditioned policy was trained by behaviour cloning and then DAgger, but without a planner it drives the arm into joint limits.

**Largest failure causes** (both modes):

- Opponents' pond tiles read below the 0.8 confidence threshold, mostly across the table.
- The claimed discard is not detected in the shared zone.
- Occasional tip-over landing errors.

## Things to know before hardware

- **Gripper direction.** The upstream i2rt finger meshes are closed at joint7 = 0; Agentic's contact surrogate assumes the opposite. Check which is true.
- **Pad width.** The model assumes 15 mm pads, measured from the upstream meshes.
- **Workspace.** It is limited by the YAM's ±10 N m joint torques and by joint 6's ±120°:
  - robot-filled slots stay within y ≤ 0.30 m;
  - an upright tile can be held face-preserving only between bearings of about 60° and 150°;
  - the whole layout is in `sim/layout.py`.
- **Tile presentation.** Draws are presented upright, facing the robot; reorientation is a staged side-regrasp roll ([ADR 0004](docs/adr/0004-tile-presentation-and-contact-skills.md)).
- **Assumed physical values.** Tile size (20 × 26 × 16 mm, 12 g), friction, and camera intrinsics (from the published fields of view) are assumptions to measure.

## Reproducing

```bash
uv sync --extra sim
uv run pytest && uv run pytest -m slow
uv run mahjong-robot train --config configs/gate1.yaml                 # about 2 h on 24 CPU cores
uv run mahjong-robot gate1 --policy runs/gate1/checkpoints/best.pt --opponents heldout_heuristic
uv run python scripts/build_tile_dataset.py                            # classifier crops
uv run python -c "from pathlib import Path; from mahjong_robot.vision.classifier import train; train(Path('data/tiles_train.npz'), Path('data/tiles_val.npz'), Path('models/tile_classifier.pt'), epochs=35)"
uv run python scripts/generate_contact_data.py data/contact.npz 140 0 16  # contact policy data
uv run python -c "from pathlib import Path; from mahjong_robot.sim.learned import train_models; train_models(Path('data/contact.npz'), Path('models/contact_policy.pt'), epochs=45)"
uv run python -c "import torch; from mahjong_robot.sim.learned import dagger_approach; net,_,_=dagger_approach(0); ck=torch.load('models/contact_policy.pt', weights_only=False); ck['approach']=net.state_dict(); ck['hybrid_refined']=['pick_flat']; torch.save(ck,'models/contact_policy.pt')"
./scripts/final_eval.sh                                                # gate 2 simulation + gate 3, all modes
```
