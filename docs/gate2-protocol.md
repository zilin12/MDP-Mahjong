# Gate 2 hardware protocol: tile operations

Run on the real YAM (linear_4310 gripper, wrist D405, overhead D435i) after the checks below. Every trial is logged, including aborted ones; nothing is re-run to replace a failure.

## Before the first trial

1. **Gripper convention.** Command joint7 = 0 and check whether the fingers close. The simulator follows the upstream meshes (closed at 0); the Agentic surrogate assumes the opposite. Record the result and fix `scene.gripper_command_for_gap` if needed.
2. **Pad width.** Measure the distal fingertip width. The close-packed extraction design assumes about 15 mm, against 20 mm tiles.
3. **Tiles.** Measure your tile set. The simulation uses 20 x 26 x 16 mm and 12 g; update `sim/layout.py` if yours differ.
4. **Camera calibration.** Calibrate the D435i extrinsic against the table and the D405 hand-eye transform. Simulation assumes errors of about 1 mm / 0.1° and 0.5 mm / 0.3° respectively.
5. **Mounting and layout.** Mount the base and mark the hand row, pond and meld slots, staging spot and shared zone as in `sim/layout.py`. Robot-filled slots are 40 mm apart, and the far pond row is at most 0.30 m from the base (joint-torque limit).
6. **Held-out split.** Collect operator-guided demonstrations first, then freeze the policy. Held-out trials use tile sets and poses never used for demonstrations.

## Trials

For each skill, run 50 held-out trials with fresh randomized placements, following the success definitions in [evaluation.md](evaluation.md).

```bash
mahjong-robot trial add --gate 2 --kind close_packed_extraction --success yes
mahjong-robot trial add --gate 2 --kind insertion --success no --reason "tile landed on neighbour edge"
mahjong-robot trial add --gate 2 --kind recognition --tile-type 12 --success yes
mahjong-robot trial report --gate 2
```

Skill kinds: `shared_zone_pick`, `close_packed_extraction`, `insertion`, `reorientation`, `placement`. For recognition, log 10 held-out camera views per tile type, with the camera's reading compared against the operator's label.
