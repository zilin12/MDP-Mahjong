# The playing agent decides by tile type; perception binds the tile instance

The playing agent outputs semantic actions over tile types (for example, "discard 5 Bamboo" or "Chow with the discard in the middle"), never tile instances, poses, or motions. It receives only its player view from the engine, which keeps the other concealed hands and wall order. A separate execution layer binds each action to a visible tile instance through perception, runs the tile operations, and reports the game event to the engine only after verified execution. On uncertainty or failure it pauses and reconciles, and it never guesses. Forced draws belong to the game orchestrator.

We chose this split because the same trained policy then runs unchanged in the symbolic engine (gate one), the physics simulator (gate three), and on hardware (gate four). It also keeps hidden-information isolation testable at one interface, so failures can be attributed to strategy, perception, or manipulation. The cost is that the agent cannot prefer one physical copy of a tile over another, for example the easier-to-reach one. Binding makes that choice, and binding never changes the game meaning.

## Consequences

- Simulator ground truth may score trials but must never reach the playing agent or binding.
- A failed operation leaves the engine state unchanged until reconciliation, so the engine and the physical table cannot silently diverge.
