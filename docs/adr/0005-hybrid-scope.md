# The hybrid system uses the learned contact policy only where development trials validate it

PLAN.md splits manipulation into planned free-space motion and a learned visuomotor contact phase. The learned wrist-camera policy (`sim/learned.py`) predicts the correction from the overhead estimate to the true contact pose. On held-out samples it is accurate for flat picks (90th-percentile residual 0.9 mm), but not for left-end insertion (5.8 mm, because the held tile hides the row end from the wrist camera) or the side regrasp (4.7 mm). In development skill trials it also did worse than open-loop classical contact for close-packed extraction (8/20 vs 20/20) and right-end insertion (0/10 vs 10/10). The simulated overhead D435i, calibrated to about 1 mm / 0.1°, is accurate enough for open-loop contact, and the learned policy's occasional large errors push neighbouring tiles.

We therefore fixed the hybrid's scope before the final runs. It refines a contact phase with the learned policy only where development trials show no loss against classical control: currently the flat pick of claimed discards. It also feeds the wrist-camera corrections back as an online bias estimate for later overhead estimates. The learned-only ablation still uses the learned policy for every contact phase and for free-space motion, and it is reported as measured.

## Consequences

- In simulation, the hybrid and classical systems differ mainly in the flat-pick contact phase, so their gate-3 results are expected to be close. That is a finding about this simulator's perception accuracy, not a property of real hardware. Gate 2's operator-guided real demonstrations should re-run this selection on hardware.
- Development and final seeds are disjoint. Gate-2 final trials use seeds from 60,000; gate-3 final hands use held-out block 1.
