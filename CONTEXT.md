# Single-arm Mahjong

Vocabulary for a project in which one YAM arm plays one seat of a simplified Chinese Mahjong hand, from game decision through physical tile manipulation.

## Game

**Hand**:
One deal played from the initial 13/14 tiles until a win or wall exhaustion. It is the only episode unit; there is no multi-hand match, dealer rotation, or score.
_Avoid_: Game, round, match

**Rule profile**:
The supplied simplified Chinese rules together with the fixed conventions that make every transition decidable (ADR 0001).
_Avoid_: Ruleset when meaning only the supplied text

**Tile type**:
One of the 34 distinct faces (1–9 in three suits, four Winds, three Dragons). Game decisions are expressed in tile types.
_Avoid_: Tile kind, face

**Tile instance**:
One of the 136 physical or simulated tiles; four instances share each tile type.
_Avoid_: Tile ID when meaning the type

**Seat**:
A player position, 0–3 in counterclockwise turn order; seat 0 is the dealer.

**Robot seat**:
The seat whose decisions come from the playing agent and, from gate three on, whose tiles are handled by the arm.

**Live wall**:
The ordered undrawn tiles. Normal draws take from its front; Kong replacement draws take from its back.
_Avoid_: Dead wall (this rule profile has none)

**Concealed hand**:
A seat's unexposed tiles. Its contents are known only to the engine and to that seat.

**Meld**:
A Chow, Pung, or Kong. Each counts as one of the four melds in a Standard Win.

**Exposed meld**:
A meld formed with a claimed discard or by adding to an exposed Pung, placed face-up.

**Concealed Kong**:
A Kong declared from four tiles in the concealed hand. It is a meld but uses no claimed tile.

**Added Kong**:
A Kong formed by adding the fourth tile from the concealed hand to one's own exposed Pung.

**Claim**:
A response to another seat's discard: win, Pung, Kong, Chow, or pass.
_Avoid_: Call when meaning the response window

**Claim window**:
The moment after a discard when non-discarding seats may claim it.

**Winning structure**:
Standard Win (four melds plus a pair), Seven Pairs, or Thirteen Orphans.

**Self-draw win**:
A win completed by a tile the winner drew, including a Kong replacement tile.

**Win by discard**:
A win completed by claiming another seat's discard.

**Wall exhaustion**:
The end of a hand in which a seat must draw and the live wall is empty.
_Avoid_: Draw (ambiguous with drawing a tile)

**Completed hand**:
A hand played to its end (any seat's win or wall exhaustion) without an aborting fault. Completion does not require the robot seat to win.

## Playing agent

**Playing agent**:
The component that chooses the robot seat's semantic action from its player view.
_Avoid_: AI, bot

**Player view**:
Everything one seat may know: its concealed hand, all public events, and the counts the table shows. It never includes other concealed hands or wall order.
_Avoid_: State, observation (when meaning engine state)

**Semantic action**:
A game action named by tile type, such as "discard 5 Bamboo" or "Chow with the discard in the middle".
_Avoid_: Move, command

**Legal-action mask**:
The set of semantic actions the rule profile permits at a decision.

**Decision point**:
A moment at which a seat has at least one choice. A claim window with only "pass" is not a decision point.

**Forced draw**:
A draw the rule profile requires; the game orchestrator performs it, never the playing agent.

**Legal random player**:
A baseline that chooses uniformly among legal semantic actions.

**Heuristic player**:
A fixed baseline that minimizes shanten and claims whenever claiming does not increase it.

**Shanten**:
The minimum number of tile exchanges needed to reach a ready hand; minus one means complete.

**Paired hand**:
Two evaluation hands that share a wall and opponent seeds and differ only in the robot-seat player.

**Seat-balanced**:
An evaluation set in which the robot seat occupies every seat equally often.

**Held-out**:
Seeds or opponents never used in training or checkpoint selection.

## Manipulation and perception

**Binding**:
Choosing which visible tile instance realises a semantic action's tile type.

**Tile operation**:
A physical skill: shared-zone pick, close-packed extraction, insertion, reorientation, or discard/meld placement.
_Avoid_: Primitive, skill (alone)

**Shared zone**:
A fixed reachable table region where humans (hardware) or the scene (simulation) present tiles the robot seat must take.

**Close-packed hand**:
The robot seat's concealed tiles standing upright and touching, with no rack slots or gaps.

**Free-space motion**:
Arm motion with no intended contact, handled by classical planning.

**Contact phase**:
The part of a tile operation in which the gripper or tile touches other tiles; handled by the learned visuomotor policy in the hybrid system.

**Hybrid system**:
Classical free-space motion plus a learned contact phase.

**Classical-only**:
The ablation in which scripted, geometry-driven motions also perform the contact phase.

**Learned-only**:
The ablation in which the learned policy performs the whole operation, including free-space approach.

**Verified execution**:
An operation whose outcome fresh perception confirms. Game state advances only after verified execution.

**Reconciliation**:
The pause in which the system compares perceived tiles with the engine's expectation after an uncertain or failed operation.

**Fault**:
A stop caused by uncertain identity, stale observation, collision risk, or failed verification. A fault that cannot be reconciled aborts the hand and counts against the gate.

**Ground truth**:
Simulator state used only to score trials, never as a system input.

## Physical play

**Human assistance**:
The bounded help allowed at gate four: dealing, playing the other three seats, signalling event timing, and presenting unreachable tiles in the shared zone.

**Corrective touch**:
Any human contact with the robot seat's tiles or arm during a trial. It fails the trial.

**Event signal**:
A human indication that a public event happened. It carries timing, never tile identity.
