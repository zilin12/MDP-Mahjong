# Gate evaluation criteria

Written before the final gate runs; the thresholds come from PLAN.md, and the operational definitions below make each one checkable. Simulation and hardware results are recorded and reported separately.

## Gate 1 — playing agent (symbolic engine)

- **Test hands:** 10,000 held-out wall seeds starting at 2,000,000,000 (training uses seeds below 10^9, checkpoint selection uses 10^9 + i). The robot seat is `seed % 4`, so each seat gets 2,500 hands.
- **Opponents:** three copies of the held-out heuristic (`heldout_heuristic`: shanten-greedy, ukeire tie-break, never Chows), which is never used in training or selection.
- **Pairing:** each hand is played three times (the policy, the legal random player, and the training heuristic in the robot seat) with the same wall and opponent seeds.
- **Pass:** zero illegal actions by the policy, and the lower bound of the 95% CI of the paired win-rate difference (policy minus random) above zero. The policy-minus-heuristic difference is reported, not required.
- **Policy:** the checkpoint with the best validation win rate against the training heuristic, evaluated greedily.
- **Accepted policy (29 September 2026):** the project owner accepted the second policy (`runs/gate1_v2`, copied to `models/playing_policy.pt`) for all later gates. It scored 29.8% [28.9, 30.7] on a fresh test block, against a 30% goal, statistically level with the heuristic's 30.5%. The first policy (`runs/gate1/checkpoints/best.pt`, 13.4%) is superseded.

## Gate 2 — tile operations

**Recognition.** 10 held-out views of each of the 34 types; at least 90% overall and at least 80% for every type. In simulation, each type gets 10 overhead face-up views and 10 wrist face-on views, rendered with seeds never used in training.

**Skills.** For each of the five skills, 50 held-out trials with at least 90% success. A trial's targets come from perception, never ground truth. A trial succeeds when:

| Skill | Success |
| --- | --- |
| Shared-zone pick | The presented tile (upright draw or face-up claimed discard) is held between both pads and lifted; no other tile moves more than 3.5 mm or topples. |
| Close-packed extraction | The chosen tile of a 13-tile close-packed row is held and lifted; no other tile moves more than 3.5 mm (the close-packed gap tolerance) or topples. |
| Insertion | The held upright tile stands against the free end of a 12-tile row with a gap of at most 6 mm (compaction closes it later), within 4 mm of the row line; no row tile moves more than 3.5 mm or topples. |
| Reorientation | An upright tile ends face-up (tip-over into a pond or meld slot). |
| Placement | A tile ends face-up within 6 mm of its pond or meld slot (tip-over placement, or a face-up claimed tile moved flat). |

The physical gate uses the same definitions, recorded with `mahjong-robot trial add --gate 2` (see [gate2-protocol.md](gate2-protocol.md)). Simulation results come from `mahjong_robot.sim.gate2_sim` and are not gate-2 evidence on their own.

## Gate 3 — physics integration

- **Test hands:** 100 held-out seeds from the 3,000,000,000 range, in blocks of 100; the robot seat is `seed % 4`. Block 1 compared the modes with the first policy; block 2 (hands 200–299) is used with the accepted policy. Opponents are three training heuristics played by the simulated table.
- **Completed hand:** it reaches a win (any seat) or wall exhaustion, with no aborting fault and no ground-truth discrepancy. After every robot discard and meld, the physical tiles must match what the robot reported, and at the end the robot's believed concealed hand must equal the engine's.
- **Faults** (each aborts the hand): uncertain identity (classifier confidence below 0.8 after one re-observation), verification failure after one reconciliation, collision risk (the joint-torque monitor proxy), planning failure, and belief mismatch (the chosen action is illegal in the engine).
- **Pass:** at least 70 of 100 completed, for the hybrid system.
- **Comparison:** classical-only and learned-only on the same 100 seeds. Report the completion rate, simulated time per completed hand, recoveries, and fault categories.

## Gate 4 — physical play

At least 3 of 5 complete hands without corrective touch, and at least 8 of 10 trials for each of Chow, Pung, exposed Kong, concealed Kong, added Kong, self-draw win and win by discard. Recorded with `mahjong-robot trial add --gate 4` (see [gate4-protocol.md](gate4-protocol.md)).
