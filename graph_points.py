"""
Plot a 2D point set together with its convex hull (scatter + polygon outline).

Requires matplotlib (`pip install matplotlib`).
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes
except ImportError as exc:  # pragma: no cover - import guard
    raise ImportError(
        "graph_points requires matplotlib. Install with: pip install matplotlib"
    ) from exc


def _to_nx2(points: Iterable) -> np.ndarray:
    """Convert an iterable of (x, y) pairs or an array shaped (n, 2) to float array (n, 2)."""
    # Prefer ``np.asarray`` for ndarray inputs so we never iterate a 0-d array (e.g.
    # ``np.array({...some dict...})``).
    if isinstance(points, np.ndarray):
        arr = np.asarray(points, dtype=float)
    else:
        arr = np.asarray(list(points), dtype=float)

    if arr.ndim == 0:
        raise ValueError(
            "Expected points shaped (n, 2). Got a 0-d array — often caused by "
            "``np.array(a_dict)`` instead of stacking ``(x, y)`` coordinates."
        )
    if arr.size == 0:
        return np.empty((0, 2))
    if arr.ndim == 1:
        if arr.shape[0] != 2:
            raise ValueError("A single point must have shape (2,) — (x, y).")
        arr = arr.reshape(1, 2)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"Expected points shaped (n, 2); got {arr.shape}.")
    return arr


def plot_points_and_hull(
    all_points: Iterable,
    hull_vertices: Iterable,
    *,
    ax: Optional[Axes] = None,
    interior_color: str = "C0",
    hull_color: str = "red",
    interior_size: float = 25.0,
    hull_size: float = 60.0,
    draw_hull_polygon: bool = True,
    hull_line_width: float = 1.5,
    equal_aspect: bool = True,
    title: Optional[str] = None,
    show: bool = True,
) -> Axes:
    """
    Scatter all points, emphasize hull vertices, and optionally stroke the hull.

    Parameters
    ----------
    all_points
        Every point (interior + hull). Each row is ``(x, y)``.
    hull_vertices
        Hull vertices in **counter-clockwise** order (typical convex-hull output).
        If ``draw_hull_polygon`` is ``False``, order only affects display if you
        rely on ``plot`` elsewhere; for the polygon outline, CCW order matters.
    ax
        Optional Matplotlib axes. If ``None``, a new figure and axes are created.
    interior_color / hull_color
        Scatter colors for the full set vs hull vertices (hull drawn on top).
    interior_size / hull_size
        Marker sizes for the two scatters.
    draw_hull_polygon
        If ``True`` and there are at least two hull vertices, draw line segments
        along ``hull_vertices`` in order; if at least three, close back to the
        first vertex.
    equal_aspect
        Use equal scaling on x and y so angles and distances read correctly.
    title
        Optional axes title.
    show
        If ``True``, call ``plt.show()`` before returning (ignored when ``ax``
        is provided by the caller).

    Returns
    -------
    matplotlib.axes.Axes
        The axes used for plotting.
    """
    pts = _to_nx2(all_points)
    hull = _to_nx2(hull_vertices)

    created_fig = ax is None
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))

    if pts.shape[0] > 0:
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            s=interior_size,
            c=interior_color,
            label="points",
            zorder=2,
            edgecolors="black",
            linewidths=0.4,
            alpha=0.85,
        )

    if hull.shape[0] > 0:
        ax.scatter(
            hull[:, 0],
            hull[:, 1],
            s=hull_size,
            c=hull_color,
            label="hull vertices",
            zorder=3,
            edgecolors="darkred",
            linewidths=0.6,
        )

    if draw_hull_polygon and hull.shape[0] >= 2:
        loop = hull
        if hull.shape[0] >= 3:
            loop = np.vstack([hull, hull[0:1]])
        ax.plot(
            loop[:, 0],
            loop[:, 1],
            color=hull_color,
            linewidth=hull_line_width,
            linestyle="-",
            zorder=1,
            label="hull" if hull.shape[0] >= 3 else "hull segment",
        )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if title:
        ax.set_title(title)
    if equal_aspect:
        ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle=":", alpha=0.5)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc="best")

    if show and created_fig:
        plt.show()

    return ax
