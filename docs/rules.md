<!-- Supplied by the project owner during planning (2026-09-28), reproduced verbatim.
     Conventions that complete this text are fixed in docs/adr/0001-rule-profile.md. -->

## Simplified Chinese Mahjong Rules

### 1. Players and Tiles

The game is played by **four players** using a standard set of **136 tiles**:

- 36 Character tiles (1–9 Characters, four copies each)
- 36 Bamboo tiles (1–9 Bamboos, four copies each)
- 36 Dot tiles (1–9 Dots, four copies each)
- 16 Wind tiles (East, South, West, and North)
- 12 Dragon tiles (Red, Green, and White)

Flower and Season tiles are excluded in this simplified rule set.

Each player begins with **13 tiles**, while the dealer begins with **14 tiles** and makes the first discard.

### 2. Basic Turn

Players take turns in counterclockwise order.

During a normal turn, a player:

1. Draws one tile from the wall.
2. Checks whether the hand can win.
3. May declare a Kong if applicable.
4. Discards one tile.

After discarding, the player normally returns to a 13-tile hand.

Play continues until a player wins or no drawable tiles remain.

### 3. Chow

A **Chow** is a sequence of three consecutive numbered tiles of the same suit.

Example:

**3 Bamboo + 4 Bamboo + 5 Bamboo**

A player may claim a discarded tile for a Chow **only from the player immediately before them in turn order**.

Honor tiles cannot form Chows.

### 4. Pung

A **Pung** consists of three identical tiles.

Example:

**7 Dots + 7 Dots + 7 Dots**

A player may claim a matching discarded tile from **any player** to complete a Pung.

The resulting Pung is placed face-up as an exposed meld.

### 5. Kong

A **Kong** consists of four identical tiles.

Example:

**Red Dragon × 4**

After declaring a Kong, the player draws a replacement tile so that the hand maintains the correct number of effective tiles.

A Kong counts as **one meld** when determining whether the hand is complete.

Kongs may be concealed, formed using another player's discard, or formed by adding a fourth tile to an existing exposed Pung.

### 6. Claim Priority

If multiple players want the same discarded tile, claims are resolved in the following priority:

**Win > Pung/Kong > Chow**

A winning claim therefore takes priority over other calls.

### 7. Winning Conditions

A player wins when their hand satisfies one of the following three structures.

#### Standard Win

A Standard Win consists of:

**Four melds + one pair**

A meld may be:

- A Chow
- A Pung
- A Kong

The pair consists of two identical tiles.

Example:

**123 Characters + 456 Characters + 789 Dots + 555 Bamboo + East East**

#### Seven Pairs

A **Seven Pairs** hand consists of seven pairs of identical tiles, for a total of 14 tiles.

Example:

**11 Characters + 33 Characters + 66 Characters + 22 Dots + 88 Dots + 44 Bamboo + East East**

The hand does not need to follow the normal four-meld-and-one-pair structure.

#### Thirteen Orphans

A **Thirteen Orphans** hand contains one copy of each of the following 13 terminal and honor tiles:

- 1 Character
- 9 Character
- 1 Bamboo
- 9 Bamboo
- 1 Dot
- 9 Dot
- East
- South
- West
- North
- Red Dragon
- Green Dragon
- White Dragon

The 14th tile must duplicate any one of these 13 tiles to form a pair.

### 8. Ways to Win

A player can win in two main ways:

**Self-Draw:** The player draws the tile that completes their winning hand.

**Win by Discard:** Another player discards the tile that completes the player's winning hand.

After a valid winning declaration, the hand ends immediately.

### 9. Drawn Game

If the drawable wall is exhausted before any player completes a valid winning hand, the hand ends in a **draw**.

### 10. Simplified Scoring

For the initial version of the game, no complex regional scoring or Fan system is required.

A hand is considered successful whenever it satisfies one of the three defined winning structures:

- Standard Win
- Seven Pairs
- Thirteen Orphans

More detailed scoring rules, such as All Pungs, Full Flush, Half Flush, or other special hands, can be added later without changing the basic game mechanics.
