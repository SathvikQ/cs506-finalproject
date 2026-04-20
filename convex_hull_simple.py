"""
Dynamic planar convex hull maintenance. This module provides:

  * Exact convex hulls via Andrew's monotone chain (O(n log n) when rebuilt)
  * An OO API for insert / delete / move w/ caching
  * A kinetic-style fast path on ``move``: if the moving point was not on the
    hull and stays strictly inside the previous hull polygon, the combinatorial
    hull is unchanged and we skip recomputation. This is not a truly kinetic data structe but should speed it up somewhat

For cases where most motion is interior this avoids many full rebuilds
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, List, Mapping, Optional, Sequence, Set, Tuple
from test_points import get_points

# ---------------------------------------------------------------------------
# Geometry primitives
# ---------------------------------------------------------------------------

EPS = 1e-12


def orient(ax: float, ay: float, bx: float, by: float, cx: float, cy: float) -> float:
    """Signed twice the area of triangle (a,b,c); >0 if c is left of directed edge a --> b."""
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)


def strictly_inside_convex_polygon(
    px: float,
    py: float,
    vertices_ccw: Sequence[Tuple[float, float]],
) -> bool:
    """
    True iff ``(px,py)`` lies strictly inside a CCW convex polygon (not on edges).
    Degenerate / fewer than 3 vertices: returns False (caller should rebuild).
    """
    n = len(vertices_ccw)
    if n < 3:
        return False
    for i in range(n):
        ax, ay = vertices_ccw[i]
        bx, by = vertices_ccw[(i + 1) % n]
        if orient(ax, ay, bx, by, px, py) <= EPS:
            return False
    return True


def monotone_chain_ids(points: Mapping[str, Tuple[float, float]]) -> List[str]:
    """
    Lower + upper hull; returns vertex ids in counter-clockwise order without
    repeating the first point at the end. Collinear boundary points are skipped
    except endpoints of segments (standard monotone chain).
    """
    if not points:
        return []
    items = sorted(points.items(), key=lambda kv: (kv[1][0], kv[1][1]))
    if len(items) == 1:
        return [items[0][0]]
    if len(items) == 2:
        return [items[0][0], items[1][0]]

    pts = [(pid, x, y) for pid, (x, y) in items]

    def cross_idx(i: int, j: int, k: int) -> float:
        return orient(pts[i][1], pts[i][2], pts[j][1], pts[j][2], pts[k][1], pts[k][2])

    lower: List[int] = []
    for i in range(len(pts)):
        while len(lower) >= 2 and cross_idx(lower[-2], lower[-1], i) <= EPS:
            lower.pop()
        lower.append(i)

    upper: List[int] = []
    for i in range(len(pts) - 1, -1, -1):
        while len(upper) >= 2 and cross_idx(upper[-2], upper[-1], i) <= EPS:
            upper.pop()
        upper.append(i)

    # Omit duplicate endpoints where chains meet
    hull_idx = lower[:-1] + upper[:-1]
    return [pts[i][0] for i in hull_idx]


@dataclass
class MutablePoint:
    """A planar point whose coordinates may change over time."""

    x: float
    y: float


class DynamicConvexHull:
    """
    Maintains a set of labeled 2D points and their convex hull under insertion,
    deletion, and coordinate updates (kinetic ``move``).

    Hull recomputation is O(n log n). Moves that cannot change the hull's
    vertex set may skip recomputation via the interior fast path described in
    the module docstring.
    """

    def __init__(self) -> None:
        self._points: Dict[str, MutablePoint] = {}
        self._hull_ids: Optional[List[str]] = None
        self._hull_vertex_set: Set[str] = set()
        self._dirty: bool = True

    # point set

    def __len__(self) -> int:
        return len(self._points)

    def __contains__(self, point_id: str) -> bool:
        return point_id in self._points

    def ids(self) -> Iterator[str]:
        return iter(self._points.keys())

    def add_point(self, point_id: str, x: float, y: float) -> None:
        if point_id in self._points:
            raise KeyError(f"Point id already exists: {point_id!r}")
        self._points[point_id] = MutablePoint(x, y)
        self._dirty = True

    def remove_point(self, point_id: str) -> None:
        if point_id not in self._points:
            raise KeyError(point_id)
        del self._points[point_id]
        self._dirty = True

    def get_point(self, point_id: str) -> Tuple[float, float]:
        p = self._points[point_id]
        return (p.x, p.y)

    def move_point(self, point_id: str, x: float, y: float) -> None:
        """
        Update coordinates. Uses a hull fast path when the moving point was not
        a hull vertex and remains strictly inside the previous hull polygon.
        """
        if point_id not in self._points:
            raise KeyError(point_id)

        was_on_hull = point_id in self._hull_vertex_set and not self._dirty
        p = self._points[point_id]
        p.x, p.y = x, y

        if self._dirty or len(self._points) < 3:
            self._rebuild_hull()
            return

        if was_on_hull:
            self._rebuild_hull()
            return

        poly = self._hull_polygon_coords()
        if poly is not None and strictly_inside_convex_polygon(x, y, poly):
            return

        self._rebuild_hull()

    def apply_moves(self, moves: Mapping[str, Tuple[float, float]]) -> None:
        """Update many coordinates and rebuild the hull once (O(n log n))."""
        for pid, (x, y) in moves.items():
            if pid not in self._points:
                raise KeyError(pid)
            p = self._points[pid]
            p.x, p.y = x, y
        self._dirty = True
        self._rebuild_hull()

    def set_points(self, points: Mapping[str, Tuple[float, float]]) -> None:
        """Replace the entire set (handy for resetting a simulation)."""
        self._points.clear()
        for pid, (x, y) in points.items():
            self._points[pid] = MutablePoint(x, y)
        self._dirty = True

    # hull query

    def touch(self) -> None:
        """Mark hull dirty (e.g. after external mutation of MutablePoint — avoid if possible)."""
        self._dirty = True

    def hull_vertex_ids(self) -> List[str]:
        """Vertices of the convex hull in CCW order; empty if no points."""
        self._ensure_hull()
        return list(self._hull_ids or [])

    def hull_vertices(self) -> List[Tuple[str, float, float]]:
        """``(id, x, y)`` for each hull vertex, CCW."""
        return [(pid, self._points[pid].x, self._points[pid].y) for pid in self.hull_vertex_ids()]

    def area(self) -> float:
        """Polygon area of the hull; 0 if fewer than 3 vertices."""
        verts = self.hull_vertices()
        if len(verts) < 3:
            return 0.0
        s = 0.0
        n = len(verts)
        for i in range(n):
            _, x1, y1 = verts[i]
            _, x2, y2 = verts[(i + 1) % n]
            s += x1 * y2 - x2 * y1
        return abs(s) * 0.5

    def extreme_in_direction(self, dx: float, dy: float) -> Optional[str]:
        """
        Return a point id that maximizes dot((x,y), (dx,dy)) over the set.
        Ties broken by smallest id string for stability (O(n) scan).
        """
        if not self._points:
            return None
        best_id: Optional[str] = None
        best_val: Optional[float] = None
        for pid, pt in sorted(self._points.items(), key=lambda kv: kv[0]):
            v = pt.x * dx + pt.y * dy
            if best_val is None or v > best_val + EPS:
                best_val = v
                best_id = pid
        return best_id

    # internals

    def _snapshot_xy(self) -> Dict[str, Tuple[float, float]]:
        return {pid: (pt.x, pt.y) for pid, pt in self._points.items()}

    def _ensure_hull(self) -> None:
        if self._dirty:
            self._rebuild_hull()

    def _rebuild_hull(self) -> None:
        snap = self._snapshot_xy()
        self._hull_ids = monotone_chain_ids(snap)
        self._hull_vertex_set = set(self._hull_ids or [])
        self._dirty = False

    def _hull_polygon_coords(self) -> Optional[List[Tuple[float, float]]]:
        self._ensure_hull()
        if not self._hull_ids or len(self._hull_ids) < 3:
            return None
        return [(self._points[pid].x, self._points[pid].y) for pid in self._hull_ids]


def batch_move(hull: DynamicConvexHull, moves: Mapping[str, Tuple[float, float]]) -> None:
    """Module-level helper; equivalent to ``hull.apply_moves(moves)``."""
    hull.apply_moves(moves)


points = list(
    map(
        lambda p: MutablePoint(p[0], p[1]),
         get_points(1000)
    ))
hull = DynamicConvexHull()
for i, p in enumerate(points):
    hull.add_point(str(i), p.x, p.y)
print(hull.hull_vertices())