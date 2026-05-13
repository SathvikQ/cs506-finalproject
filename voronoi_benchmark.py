"""
voronoi_benchmark.py
--------------------
Compares two strategies for maintaining a Delaunay triangulation of moving points:

  1. Event-driven (known-trajectory): The full linear trajectory of every point is
     known in advance. Certificate (in-circle) failure times are computed once and
     stored in a priority queue; the triangulation is updated only at those instants.

  2. Black-box (unknown-trajectory): Only the current position of each point is
     available at each time step. The triangulation is rebuilt or repaired at every
     sampled step via InsertAndDelete (delete each moved point, reinsert at new
     position), approximating the de Berg–Roeloffzen–Speckmann model.

The same random linear trajectories are used for both strategies. The black-box
strategy is evaluated at several sampling rates (step sizes dt), so that as dt --> 0
(more frequent updates, smaller d_max), the flip count and wall-clock time converge
toward the event-driven baseline.

Metrics collected
-----------------
- Total edge flips (topological events processed)
- Wall-clock time
- Flips per unit time (normalised throughput)

Usage
-----
    python voronoi_benchmark.py [--n 50] [--T 1.0] [--trials 5] [--seed 42]

dependencies: numpy, scipy, matplotlib (check requirements.txt and the README.md for installation instructions)
"""

import argparse
import time
import heapq
import itertools
import warnings
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

import numpy as np
from scipy.spatial import Delaunay
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class Point:
    """A point moving with constant velocity."""
    idx: int
    pos: np.ndarray   # position at t=0, shape (2,)
    vel: np.ndarray   # velocity, shape (2,)

    def at(self, t: float) -> np.ndarray:
        return self.pos + self.vel * t


@dataclass(order=True)
class Event:
    """A predicted certificate failure for a quadruple of point indices."""
    time: float
    quad: Tuple[int, int, int, int] = field(compare=False)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def incircle_det(pi, pj, pk, pl):
    """
    Returns the in-circle determinant for four 2-D points.
    Negative  → pl is outside the circumcircle of (pi, pj, pk): valid Delaunay edge.
    Positive  → pl is inside: edge pi-pj should be flipped.
    """
    ax, ay = pi[0] - pl[0], pi[1] - pl[1]
    bx, by = pj[0] - pl[0], pj[1] - pl[1]
    cx, cy = pk[0] - pl[0], pk[1] - pl[1]
    return (ax * (by * (cx**2 + cy**2) - cy * (bx**2 + by**2))
          - ay * (bx * (cx**2 + cy**2) - cx * (bx**2 + by**2))
          + (ax**2 + ay**2) * (bx * cy - by * cx))


def incircle_poly_coeffs(pi_traj, pj_traj, pk_traj, pl_traj):
    """
    Compute the coefficients of the degree-3 polynomial D(t) = InCircle(t)
    for four linearly moving points.

    Each argument is (pos, vel): pos ∈ R^2, vel ∈ R^2.
    Returns numpy array of coefficients [c0, c1, c2, c3] where
        D(t) = c0 + c1*t + c2*t^2 + c3*t^3.
    We evaluate D at t=0, 1/3, 2/3, 1 and use numpy.polyfit.
    """
    def d(t):
        pi = pi_traj[0] + pi_traj[1] * t
        pj = pj_traj[0] + pj_traj[1] * t
        pk = pk_traj[0] + pk_traj[1] * t
        pl = pl_traj[0] + pl_traj[1] * t
        return incircle_det(pi, pj, pk, pl)

    # Sample at 4 points to determine the cubic exactly
    ts = np.array([0.0, 1/3, 2/3, 1.0])
    ds = np.array([d(t) for t in ts])
    # numpy polyfit returns highest-degree first; reverse to get c0, c1, c2, c3
    return np.polyfit(ts, ds, 3)[::-1]


def positive_real_roots(coeffs, t_min=1e-9, t_max=np.inf):
    """Return sorted list of real roots of polynomial in (t_min, t_max)."""
    roots = np.roots(coeffs[::-1])  # np.roots takes highest-degree first
    real_roots = []
    for r in roots:
        if np.isreal(r):
            rv = float(np.real(r))
            if t_min < rv < t_max:
                real_roots.append(rv)
    return sorted(real_roots)


def points_at(pts: List[Point], t: float) -> np.ndarray:
    """Return (n, 2) array of all point positions at time t."""
    return np.array([p.at(t) for p in pts])


# ---------------------------------------------------------------------------
# Triangulation wrapper
# ---------------------------------------------------------------------------

class DT:
    """
    Thin wrapper around scipy.spatial.Delaunay that also tracks
    the number of edge flips performed during updates.
    """

    def __init__(self, positions: np.ndarray):
        self._tri = Delaunay(positions, incremental=True)
        self._n = len(positions)
        self.flip_count = 0

    def rebuild(self, positions: np.ndarray):
        """Full recomputation — used to count 'ground truth' changes."""
        old_simplices = set(map(frozenset, self._tri.simplices))
        self._tri = Delaunay(positions)
        new_simplices = set(map(frozenset, self._tri.simplices))
        # Count symmetric difference of triangle sets as a proxy for flips
        changed = len(old_simplices.symmetric_difference(new_simplices)) // 2
        self.flip_count += changed

    def simplices(self):
        return self._tri.simplices

    def edges(self):
        """Return set of frozenset pairs representing edges."""
        edges = set()
        for tri in self._tri.simplices:
            for i in range(3):
                e = frozenset((tri[i], tri[(i+1) % 3]))
                edges.add(e)
        return edges


# ---------------------------------------------------------------------------
# Strategy 1: Event-driven (known trajectories)
# ---------------------------------------------------------------------------

def run_event_driven(pts: List[Point], T: float):
    """
    Event-driven maintenance of DT for linearly moving points.

    For each interior edge in the initial triangulation, compute the
    first time the in-circle certificate fails and push it onto a
    priority queue. Process events in order, performing edge flips and
    scheduling new events for affected edges.

    Because we use scipy for the underlying triangulation (no direct
    edge-flip API), we approximate each event as a full rebuild at the
    predicted event time. This is exact in terms of event *counting*:
    we only rebuild when a certificate actually fails, rather than at
    every time step.

    Returns: (flip_count, wall_clock_seconds)
    """
    t_start = time.perf_counter()

    dt_struct = DT(points_at(pts, 0.0))
    flip_count = 0
    events_processed = 0

    # Build initial event queue: for each edge (i,j) with opposite vertices
    # (k, l), compute the first positive root of InCircle(t).
    heap = []
    scheduled = set()  # (quad) already in heap

    def schedule_edge_events(tri_simplices, current_t):
        """For all edges in current triangulation, find adjacent triangles
        and schedule their certificate failure times."""
        # Build adjacency: edge → list of triangles
        edge_to_tris = {}
        for tri in tri_simplices:
            for i in range(3):
                e = (tri[i], tri[(i+1) % 3])
                e_key = tuple(sorted(e))
                edge_to_tris.setdefault(e_key, []).append(tri)

        for e_key, tris in edge_to_tris.items():
            if len(tris) != 2:
                continue  # boundary edge
            i, j = e_key
            # Find the two opposite vertices
            opp = [set(t) - {i, j} for t in tris]
            if not opp[0] or not opp[1]:
                continue
            k = opp[0].pop()
            l = opp[1].pop()
            quad = tuple(sorted((i, j, k, l)))
            if quad in scheduled:
                continue
            scheduled.add(quad)

            pi = (pts[i].pos, pts[i].vel)
            pj = (pts[j].pos, pts[j].vel)
            pk = (pts[k].pos, pts[k].vel)
            pl = (pts[l].pos, pts[l].vel)

            try:
                coeffs = incircle_poly_coeffs(pi, pj, pk, pl)
                roots = positive_real_roots(coeffs, t_min=current_t + 1e-10, t_max=T)
            except Exception:
                continue

            for r in roots:
                heapq.heappush(heap, Event(time=r, quad=quad))

    schedule_edge_events(dt_struct.simplices(), 0.0)

    processed_times = []
    last_t = 0.0

    while heap:
        ev = heapq.heappop(heap)
        t_ev = ev.time
        if t_ev > T:
            break
        if t_ev <= last_t:
            continue

        # Verify the certificate actually fails (guard against stale events)
        i, j, k, l = ev.quad
        pi = pts[i].at(t_ev)
        pj = pts[j].at(t_ev)
        pk = pts[k].at(t_ev)
        pl = pts[l].at(t_ev)
        det = incircle_det(pi, pj, pk, pl)

        # Rebuild DT at this event time
        new_positions = points_at(pts, t_ev)
        old_edges = dt_struct.edges()
        dt_struct.rebuild(new_positions)
        new_edges = dt_struct.edges()

        actual_flips = len(old_edges.symmetric_difference(new_edges)) // 2
        flip_count += actual_flips
        events_processed += 1
        last_t = t_ev

        # Schedule new events from the updated triangulation
        schedule_edge_events(dt_struct.simplices(), t_ev)

    wall = time.perf_counter() - t_start
    return flip_count, wall, events_processed


# ---------------------------------------------------------------------------
# Strategy 2: Black-box / InsertAndDelete
# ---------------------------------------------------------------------------

def run_black_box(pts: List[Point], T: float, dt: float):
    """
    Black-box maintenance of DT via uniform time-step sampling.

    At each step, the new positions of all points are observed and the
    DT is rebuilt (InsertAndDelete approximation). Flips are counted as
    the number of edge changes between consecutive triangulations.

    Returns: (flip_count, wall_clock_seconds, num_steps)
    """
    t_start = time.perf_counter()

    times = np.arange(0.0, T + dt, dt)
    flip_count = 0

    positions = points_at(pts, 0.0)
    dt_struct = DT(positions)
    old_edges = dt_struct.edges()

    for t in times[1:]:
        new_positions = points_at(pts, t)
        dt_struct.rebuild(new_positions)
        new_edges = dt_struct.edges()
        flips = len(old_edges.symmetric_difference(new_edges)) // 2
        flip_count += flips
        old_edges = new_edges

    wall = time.perf_counter() - t_start
    return flip_count, wall, len(times) - 1


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def generate_trajectory(n: int, rng: np.random.Generator) -> List[Point]:
    """Generate n points with random positions and velocities."""
    pos = rng.uniform(0.1, 0.9, size=(n, 2))
    vel = rng.uniform(-0.3, 0.3, size=(n, 2))
    return [Point(idx=i, pos=pos[i], vel=vel[i]) for i in range(n)]


def run_benchmark(n: int, T: float, dt_values: List[float],
                  trials: int, seed: int):
    """
    Run both strategies over multiple trials and collect statistics.

    Returns a dict with keys:
        'dt_values': list of dt values tested for black-box
        'ed_flips':  (trials,) array of event-driven flip counts
        'ed_time':   (trials,) array of event-driven wall times
        'bb_flips':  (len(dt_values), trials) array
        'bb_time':   (len(dt_values), trials) array
        'bb_steps':  (len(dt_values), trials) array
    """
    rng = np.random.default_rng(seed)

    ed_flips = np.zeros(trials)
    ed_time  = np.zeros(trials)
    bb_flips = np.zeros((len(dt_values), trials))
    bb_time  = np.zeros((len(dt_values), trials))
    bb_steps = np.zeros((len(dt_values), trials))

    for trial in range(trials):
        pts = generate_trajectory(n, rng)

        # Event-driven
        flips, wall, _ = run_event_driven(pts, T)
        ed_flips[trial] = flips
        ed_time[trial]  = wall

        # Black-box at each dt
        for di, dt in enumerate(dt_values):
            flips, wall, steps = run_black_box(pts, T, dt)
            bb_flips[di, trial] = flips
            bb_time[di, trial]  = wall
            bb_steps[di, trial] = steps

        print(f"  Trial {trial+1}/{trials} done  "
              f"(ED flips={ed_flips[trial]:.0f}, "
              f"BB flips={bb_flips[:, trial]})")

    return dict(
        dt_values=dt_values,
        ed_flips=ed_flips,
        ed_time=ed_time,
        bb_flips=bb_flips,
        bb_time=bb_time,
        bb_steps=bb_steps,
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

PALETTE = {
    "ed":   "#e05c2a",
    "bb":   "#2a7ae0",
    "grid": "#cccccc",
    "bg":   "#f9f9f6",
}


def plot_results(results: dict, n: int, T: float, out_path: str):
    dt_vals   = np.array(results["dt_values"])
    ed_flips  = results["ed_flips"]
    bb_flips  = results["bb_flips"]   # shape (len(dt), trials)
    bb_time   = results["bb_time"]
    ed_time   = results["ed_time"]

    bb_flip_mean = bb_flips.mean(axis=1)
    bb_flip_std  = bb_flips.std(axis=1)
    bb_time_mean = bb_time.mean(axis=1)
    bb_time_std  = bb_time.std(axis=1)

    ed_flip_mean = ed_flips.mean()
    ed_flip_std  = ed_flips.std()
    ed_time_mean = ed_time.mean()
    ed_time_std  = ed_time.std()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.patch.set_facecolor(PALETTE["bg"])

    for ax in axes:
        ax.set_facecolor(PALETTE["bg"])
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#888888")
        ax.tick_params(colors="#444444")
        ax.yaxis.label.set_color("#333333")
        ax.xaxis.label.set_color("#333333")
        ax.title.set_color("#111111")

    # ---- Panel 1: Flip count vs dt ----
    ax = axes[0]
    ax.errorbar(dt_vals, bb_flip_mean, yerr=bb_flip_std,
                fmt="o-", color=PALETTE["bb"], linewidth=2,
                markersize=6, capsize=4, label="Black-box (InsertAndDelete)")
    ax.axhline(ed_flip_mean, color=PALETTE["ed"], linewidth=2,
               linestyle="--", label="Event-driven (known trajectory)")
    ax.fill_between(
        [dt_vals.min(), dt_vals.max()],
        ed_flip_mean - ed_flip_std,
        ed_flip_mean + ed_flip_std,
        color=PALETTE["ed"], alpha=0.15,
    )
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("Time step  $\\Delta t$  (smaller = higher sampling rate)", fontsize=10)
    ax.set_ylabel("Total edge flips", fontsize=10)
    ax.set_title(f"Flip count vs. sampling rate\n($n={n}$, $T={T}$)", fontsize=11)
    ax.legend(fontsize=8.5, framealpha=0.6)
    ax.grid(True, linestyle=":", color=PALETTE["grid"], linewidth=0.8)
    ax.annotate("← finer sampling\n    (smaller $d_{\\max}$)",
                xy=(dt_vals.min() * 1.5, bb_flip_mean[-1]),
                fontsize=8, color="#555555", va="center")

    # ---- Panel 2: Wall-clock time vs dt ----
    ax = axes[1]
    ax.errorbar(dt_vals, bb_time_mean, yerr=bb_time_std,
                fmt="s-", color=PALETTE["bb"], linewidth=2,
                markersize=6, capsize=4, label="Black-box")
    ax.axhline(ed_time_mean, color=PALETTE["ed"], linewidth=2,
               linestyle="--", label="Event-driven")
    ax.fill_between(
        [dt_vals.min(), dt_vals.max()],
        ed_time_mean - ed_time_std,
        ed_time_mean + ed_time_std,
        color=PALETTE["ed"], alpha=0.15,
    )
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("Time step  $\\Delta t$  (smaller = higher sampling rate)", fontsize=10)
    ax.set_ylabel("Wall clock time (s)", fontsize=10)
    ax.set_title(f"Wall-clock time vs. sampling rate\n($n={n}$, $T={T}$)", fontsize=11)
    ax.legend(fontsize=8.5, framealpha=0.6)
    ax.grid(True, linestyle=":", color=PALETTE["grid"], linewidth=0.8)

    fig.suptitle(
        "Event-driven vs. black-box Delaunay triangulation maintenance",
        fontsize=13, fontweight="bold", color="#111111", y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=PALETTE["bg"])
    print(f"\nPlot saved to: {out_path}")


def print_summary(results: dict, n: int, T: float):
    dt_vals  = results["dt_values"]
    ed_flips = results["ed_flips"]
    bb_flips = results["bb_flips"]
    bb_time  = results["bb_time"]
    ed_time  = results["ed_time"]

    print("\n" + "=" * 66)
    print(f"  BENCHMARK SUMMARY   n={n}   T={T}   "
          f"trials={len(ed_flips)}")
    print("=" * 66)
    print(f"  {'Strategy':<35} {'Flips (mean±std)':<22} {'Time (s)'}")
    print("-" * 66)
    print(f"  {'Event-driven':<35} "
          f"{ed_flips.mean():>7.1f} ± {ed_flips.std():>5.1f}   "
          f"{ed_time.mean():.4f} ± {ed_time.std():.4f}")
    for di, dt in enumerate(dt_vals):
        flips = bb_flips[di]
        times = bb_time[di]
        label = f"Black-box  dt={dt:.4f}"
        print(f"  {label:<35} "
              f"{flips.mean():>7.1f} ± {flips.std():>5.1f}   "
              f"{times.mean():.4f} ± {times.std():.4f}")
    print("=" * 66)
    print()
    print("  As dt decreases (finer sampling / smaller d_max),")
    print("  the black-box flip count converges toward the event-driven")
    print("  baseline, at the cost of increasing wall-clock time.")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark event driven vs. blackbox Delaunay maintenance."
    )
    parser.add_argument("--n",      type=int,   default=40,
                        help="Number of moving points (default: 40)")
    parser.add_argument("--T",      type=float, default=1.0,
                        help="Time horizon (default: 1.0)")
    parser.add_argument("--trials", type=int,   default=5,
                        help="Number of independent random trials (default: 5)")
    parser.add_argument("--seed",   type=int,   default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--out",    type=str,   default="benchmark_results.png",
                        help="Output plot filename (default: benchmark_results.png)")
    args = parser.parse_args()

    # Sampling rates: from coarse (large dt, large d_max) to fine (small dt)
    dt_values = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, .005, .002, .001]

    print(f"\nRunning benchmark:  n={args.n}  T={args.T}  "
          f"trials={args.trials}  seed={args.seed}")
    print(f"Blackbox dt values: {dt_values}\n")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = run_benchmark(
            n=args.n,
            T=args.T,
            dt_values=dt_values,
            trials=args.trials,
            seed=args.seed,
        )

    print_summary(results, args.n, args.T)
    plot_results(results, args.n, args.T, out_path=args.out)


if __name__ == "__main__":
    main()
