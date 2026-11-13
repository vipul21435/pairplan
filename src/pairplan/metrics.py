"""Small ranking metrics."""

from __future__ import annotations

from collections.abc import Sequence


def kendall_tau(order_a: Sequence[int], order_b: Sequence[int]) -> float:
    """Kendall's tau between two rankings of the same items (no ties).

    Each argument lists the items from first to last. The value is
    ``1 - 4 * discordant / (m * (m - 1))``: 1 for identical orders, -1 for reversed.
    """
    if sorted(order_a) != sorted(order_b):
        raise ValueError("the two rankings must contain the same items")
    m = len(order_a)
    if m < 2:
        return 1.0
    position = {item: rank for rank, item in enumerate(order_b)}
    ranks = [position[item] for item in order_a]
    discordant = sum(1 for x in range(m) for y in range(x + 1, m) if ranks[x] > ranks[y])
    return 1.0 - 4.0 * discordant / (m * (m - 1))
