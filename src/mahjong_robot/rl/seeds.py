"""Disjoint wall-seed ranges. Test seeds are held out from training and checkpoint selection."""

TRAIN_BASE = 0  # training hands: [0, 1e9)
VAL_BASE = 1_000_000_000  # checkpoint-selection hands
TEST_BASE = 2_000_000_000  # gate evaluation hands (held-out)
SIM_TEST_BASE = 3_000_000_000  # gate-three physics hands (held-out)


def train_seeds(iteration: int, n: int) -> list[int]:
    start = TRAIN_BASE + iteration * 100_000
    assert n <= 100_000 and start + n < VAL_BASE
    return list(range(start, start + n))


def val_seeds(n: int) -> list[int]:
    return list(range(VAL_BASE, VAL_BASE + n))


def test_seeds(n: int, base: int = TEST_BASE) -> list[int]:
    return list(range(base, base + n))
