# Gate 4 hardware protocol: human-assisted physical play

Humans deal, play the other three seats, give event signals, and present tiles in the shared zone (ADR 0003). The robot's cameras alone identify tiles, and the robot alone chooses and executes its actions.

## Roles and presentation (ADR 0003, ADR 0004)

- **Deal.** Stand the robot's 13 tiles upright and close-packed, faces toward the robot, with the right end at the marked anchor.
- **Draws, including Kong replacements.** Stand the tile upright in the shared zone, face toward the robot.
- **Claimed discards.** When the robot's claim wins, move the discard face-up to the shared zone.
- **Other seats.** Place their discards face-up in their pond grids, 28 x 32 mm apart and row by row; lay their melds face-up in their meld strips.
- **Event signals.** Announce timing only: "your turn", "seat k discarded", "seat k called", and "added / concealed Kong". Never announce a tile's identity.
- **No corrective touch.** Do not touch the robot seat's tiles or the arm. Any such touch fails the trial; stop and log it.

## Trials

- **Complete hands.** Five hands from randomly shuffled walls. A hand is complete when it reaches any seat's win or wall exhaustion without an aborting fault and without corrective touch. The robot need not win.
- **Rare-action trials.** Ten each of Chow, Pung, exposed Kong, concealed Kong, added Kong, self-draw win and win by discard. Humans arrange the table so the action is legal; the robot must choose it and execute it.

```bash
mahjong-robot trial add --gate 4 --kind hand --success yes
mahjong-robot trial add --gate 4 --kind pung --success no --reason "meld tile landed face-down"
mahjong-robot trial add --gate 4 --kind hand --success yes --corrective-touch   # counts as a failure
mahjong-robot trial report --gate 4
```

**Stops.** Uncertain identity, stale observation, collision risk or failed verification stop execution and count against the gate (PLAN.md).
