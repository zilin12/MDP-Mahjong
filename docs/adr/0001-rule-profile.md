# Fixed rule profile that completes the supplied simplified rules

The supplied rules ([rules.md](../rules.md)) leave several transitions undecided, and an agent cannot be trained or benchmarked against an undecidable engine. During planning (2026-09-28) the project owner chose a complete, fixed profile over a minimal subset or an external engine. The engine implements exactly the conventions below; any change requires a new ADR and invalidates earlier gate-one results.

## Conventions

1. **Scope.** One hand is one episode: no scoring, fan, flowers, dealer rotation, or match.
2. **Seating.** Seats 0–3 in counterclockwise turn order; seat 0 is the dealer and starts with 14 tiles.
3. **Wall.** A shuffled 136-tile live wall with no reserved dead wall. Normal draws come from the front and Kong replacement draws from the back.
4. **Turn after a draw** (including the dealer's first turn and a Kong replacement draw): the seat may declare a self-draw win, a Concealed Kong, or an Added Kong, or it discards.
5. **Chow and Pung callers** expose the meld and must discard next. They do not draw, and they may not declare a Kong or a win before that discard.
6. **Kong callers** (exposed Kong from a discard, Concealed Kong, Added Kong) draw a replacement from the back of the live wall, then continue under rule 4. A Kong is illegal when the live wall is empty.
7. **No robbing the Kong.** An Added Kong cannot be claimed for a win.
8. **Claim priority.** Win > Pung/Kong > Chow. Only the next seat after the discarder may Chow. When several seats claim a win on one discard, the single winner is the claimant nearest after the discarder in turn order. Four copies per type make Pung/Kong conflicts impossible.
9. **Win optionality.** A seat may decline a win, and declining carries no restriction. Every claim window allows passing.
10. **Seven Pairs** may count four identical tiles as two pairs.
11. **Special hands** (Seven Pairs, Thirteen Orphans) require no melds of any kind, including Concealed Kongs.
12. **Wall exhaustion.** The hand ends when a seat must draw and the live wall is empty. The last drawn tile may be won on, and the last discard may be claimed for a win, Pung, or Chow, but not a Kong.
13. **Concealed Kongs are laid face-up**, so their type is public; only their unclaimed status distinguishes them from exposed Kongs. This keeps every declared meld identifiable by the overhead camera.
