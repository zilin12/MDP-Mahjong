# Single-arm Mahjong system

## Summary

Build four gated milestones: a playing agent, isolated tile manipulation, integrated play in a physics simulator, and human-assisted physical play. Use the supplied simplified Chinese rules for one 136-tile hand. The system uses one YAM arm with a linear_4310 gripper, a wrist D405, and an overhead D435i. Its primary claim is measured end-to-end reliability.

## Architecture and interfaces

- Create a fixed rules engine covering the three win structures, calls, claim priority, nearest-player resolution of simultaneous wins, and the agreed Kong and seven-pairs conventions. Keep concealed tiles and wall order inside the engine; expose only the robot’s hand, public events, and legal actions to its policy.
- Train a legal-action-masked recurrent PPO policy through fixed-opponent curriculum and saved-policy self-play. The policy chooses semantic actions by tile **type**; perception and execution bind that choice to a visible physical tile **instance**. The game orchestrator handles forced draws.
- Use calibrated RGB-D observations to identify tiles and estimate poses. Plan free-space motion and use a learned visuomotor policy for close-packed contact skills, following the split described in [SkipVLA](https://arxiv.org/abs/2609.20648). Advance game state only after visually verified execution; pause and reconcile uncertain or failed actions.
- Selectively reuse YAM models and geometry components from the Agentic workspace after checking model provenance, license, limits, gripper geometry, and camera calibration. Keep simulator ground truth available for evaluation only.
- Create a project glossary and short ADRs for the rule profile, the strategy-to-manipulation boundary, and the human-assistance boundary before implementation.

## Gates and tests

1. **Playing agent:** Unit-test tile conservation, legal transitions, calls, special wins, and hidden-information isolation. Require no illegal actions and a win-rate advantage over legal random play across 10,000 held-out, paired, seat-balanced hands, with the 95% confidence interval for the advantage above zero. Report the fixed-heuristic comparison.
2. **Tile operations:** Train with simulated and operator-guided real demonstrations. Test recognition on 10 held-out views of each of the 34 tile types: at least 90% overall and 80% for every type. Require at least 90% success over 50 held-out physical trials **per skill**: shared-zone pick, close-packed extraction, insertion, reorientation, and discard or meld placement.
3. **Physics integration:** Run the full decision-to-execution loop from simulated camera observations. Require at least 70 completed hands in 100 held-out hands. Compare the hybrid system with classical-only and learned-only manipulation on matched scenes; report success, time, recovery, and failure categories.
4. **Physical play:** Humans deal, play the other three seats, signal event timing, and present otherwise unreachable tiles in a fixed zone. Cameras supply tile identity; people do not choose robot actions or correct its hand. Require at least 3 of 5 complete hands without corrective human touch, plus at least 8 of 10 trials for each Chow, Pung, Kong form, and win route.

## Assumptions and controls

A completed hand may end in the robot’s win, another player’s win, or wall exhaustion; physical completion does not require the robot to win. Uncertain identity, stale observations, collision risk, or failed verification stop execution and count against the relevant gate. Record raw trials and failure reasons separately for simulation and hardware.

The `grill-with-docs` workflow calls for writing resolved terms and decisions as they arise. Plan Mode prevented file edits, so those documents are the first execution deliverable.
