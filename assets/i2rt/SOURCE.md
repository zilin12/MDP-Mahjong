# Vendored I2RT robot models

- Source: https://github.com/i2rt-robotics/i2rt.git, revision 5b72c47239bd056d0fa6c1a39edeb0537c89443c (v1.3.5). Copied from the Agentic workspace checkout (`Simulator/external/i2rt`), which had a clean `git status`.
- License: MIT, Copyright (c) I2RT Robotics ([LICENSE](LICENSE)).
- Files: `yam/yam.xml` and the base/link1–link5 STLs from `robot_models/arm/yam/v1`; `linear_4310/linear_4310.xml` and its STLs from `robot_models/gripper/linear_4310`.
- Modification: `meshdir` changed from `assets` to `.`. Nothing else was changed.
- Units: metres, kilograms, radians, as upstream.

## Composition (`src/mahjong_robot/sim/scene.py`)

- The gripper is merged into the arm's terminal `gripper` body, following Agentic's `yam_model.py`.
- Position actuators use i2rt `yam_v1.yml` gains (kp 80/80/80/10/10/10 with joint damping 5/5/5/1.5/1.5/1.5) and ±10 N m force limits, with gravity compensation routed through the actuators.
- Box contact surrogates replace the convex-hull finger collision meshes, which fill the space between the fingers (Agentic defect register D-021/D-022). The pads are 15 mm wide, the measured distal fingertip width.
- Two discrepancies need checking on hardware:
  - The upstream visual meshes are closed at joint7 = 0, but Agentic's surrogate is open at 0. This project follows the meshes.
  - The wrist D405 extrinsic comes from i2rt's `yam_station_linear_4310_d405` model. There is no calibrated D435i or D405 intrinsic file anywhere; intrinsics are derived from the published fields of view.
