import numpy as np


def get_points(n: int, low: float = 0.0, high: float = 1.0) -> np.ndarray:
    """Return ``n`` points in ``[low, high)`` for both x and y (shape ``(n, 2)``)."""
    rng = np.random.default_rng()
    return rng.uniform(low, high, size=(n, 2))