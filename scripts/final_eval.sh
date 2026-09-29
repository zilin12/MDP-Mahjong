#!/usr/bin/env bash
# Final simulation evaluations on held-out seeds (gate 2 sim counterpart, gate 3 all three modes).
# Gate 3 uses held-out block 2 (hands 200-299 of the SIM_TEST range). Block 0 held a preliminary
# classical run and block 1 the comparison with the first policy; both are reported separately.
set -euo pipefail
export MUJOCO_GL=egl
POLICY=models/playing_policy.pt  # accepted Gate 1 policy (runs/gate1_v2, 29.8% vs held-out heuristic)
uv run python -c "
from mahjong_robot.sim.gate2_sim import run, FINAL_SEED_BASE
import json
print(json.dumps(run('reports/gate2', modes=('classical', 'hybrid', 'learned'), n=50, processes=20, seed_base=FINAL_SEED_BASE), indent=1))
"
for mode in hybrid classical learned; do
  uv run mahjong-robot sim --policy "$POLICY" --mode "$mode" --hands 100 --start 200 --out reports/gate3/final_v2 --processes 20
done
