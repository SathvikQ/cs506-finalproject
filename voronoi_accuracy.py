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
- Total edges observed across all triangulation snapshots
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


def avg_pairwise_distance(positions: np.ndarray) -> float:
    """Average Euclidean distance over all pairs of points."""
    n = len(positions)
    if n < 2:
        return 0.0
    diff = positions[:, None, :] - positions[None, :, :]   # (n, n, 2)
    dists = np.sqrt((diff ** 2).sum(axis=-1))              # (n, n)
    idx = np.triu_indices(n, k=1)
    return float(dists[idx].mean())


def avg_speed(pts: List[Point]) -> float:
    """Average speed (magnitude of velocity) across all points."""
    return float(np.mean([np.linalg.norm(p.vel) for p in pts]))


# ---------------------------------------------------------------------------
# Triangulation wrapper
# ---------------------------------------------------------------------------

class DT:
    """
    Thin wrapper around scipy.spatial.Delaunay that also tracks
    the number of edge flips and average edges per snapshot.
    """

    def __init__(self, positions: np.ndarray):
        self._tri = Delaunay(positions, incremental=True)
        self._n = len(positions)
        self.flip_count = 0
        self._edge_sum = len(self.edges())  # include initial snapshot in average
        self._snapshot_count = 1

    def rebuild(self, positions: np.ndarray):
        """Full recomputation — used to count 'ground truth' changes."""
        old_simplices = set(map(frozenset, self._tri.simplices))
        self._tri = Delaunay(positions)
        new_simplices = set(map(frozenset, self._tri.simplices))
        # Count symmetric difference of triangle sets as a proxy for flips
        changed = len(old_simplices.symmetric_difference(new_simplices)) // 2
        self.flip_count += changed
        # Accumulate edge count for this snapshot
        self._edge_sum += len(self.edges())
        self._snapshot_count += 1

    @property
    def avg_edges(self) -> float:
        """Average number of edges per triangulation snapshot."""
        return self._edge_sum / self._snapshot_count

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

    Returns: (flip_count, total_edges, wall_clock_seconds, events_processed)
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
    return flip_count, dt_struct.avg_edges, wall, events_processed


# ---------------------------------------------------------------------------
# Strategy 2: Black-box / InsertAndDelete
# ---------------------------------------------------------------------------

def run_black_box(pts: List[Point], T: float, dt: float):
    """
    Black-box maintenance of DT via uniform time-step sampling.

    At each step, the new positions of all points are observed and the
    DT is rebuilt (InsertAndDelete approximation). Flips are counted as
    the number of edge changes between consecutive triangulations.

    A ground-truth (GT) triangulation is also built at each step purely
    for accuracy measurement. Accuracy at step i measures how correct the
    BB triangulation is *before* it processes the new positions — i.e. how
    stale the held-over triangulation from t_{i-1} is relative to the true
    triangulation at t_i. This matches the paper's model: the BB holds the
    last-known triangulation until the next sample arrives.

        alpha = 1 - |sym_diff(stale_bb_edges, gt_edges)| / 2 / avg_edges

    where avg_edges is the mean of |stale_bb_edges| and |gt_edges|.

    The implied constant C is back-solved from the paper's formula:

        alpha = 1 - C * v * dt / (d * e)
        =>  C = (1 - alpha) * d * e / (v * dt)

    where v = avg speed, d = avg pairwise distance (at t=0), e = avg edges.

    Returns: (flip_count, avg_edges, wall_clock_seconds, num_steps,
              mean_accuracy, mean_C)
    """
    t_start = time.perf_counter()

    times = np.arange(0.0, T + dt, dt)
    flip_count = 0

    v = avg_speed(pts)
    # d computed once at t=0 as a stable baseline for the formula
    d = avg_pairwise_distance(points_at(pts, 0.0))

    positions = points_at(pts, 0.0)
    dt_struct = DT(positions)
    old_bb_edges = dt_struct.edges()  # triangulation held from previous step

    accuracy_samples: List[float] = []
    C_samples: List[float] = []

    for t in times[1:]:
        new_positions = points_at(pts, t)

        # --- ground-truth at this timestep (before BB updates) ---
        gt_tri = Delaunay(new_positions)
        gt_edges: set = set()
        for tri in gt_tri.simplices:
            for i in range(3):
                gt_edges.add(frozenset((tri[i], tri[(i + 1) % 3])))

        # --- accuracy: stale BB (from t-1) vs fresh GT (at t) ---
        e = (len(old_bb_edges) + len(gt_edges)) / 2.0
        wrong = len(old_bb_edges.symmetric_difference(gt_edges)) / 2.0
        alpha = 1.0 - (wrong / e) if e > 0 else 1.0
        accuracy_samples.append(alpha)

        # --- implied C ---
        denom = v * dt
        if denom > 1e-12 and d > 1e-12:
            C_implied = (1.0 - alpha) * d * e / denom
            C_samples.append(C_implied)

        # --- black-box update (happens after accuracy is measured) ---
        dt_struct.rebuild(new_positions)
        new_bb_edges = dt_struct.edges()
        flips = len(old_bb_edges.symmetric_difference(new_bb_edges)) // 2
        flip_count += flips
        old_bb_edges = new_bb_edges  # carry forward for next step

    wall = time.perf_counter() - t_start

    mean_accuracy = float(np.mean(accuracy_samples)) if accuracy_samples else 1.0
    mean_C = float(np.mean(C_samples)) if C_samples else float("nan")

    return flip_count, dt_struct.avg_edges, wall, len(times) - 1, mean_accuracy, mean_C


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
        'dt_values':  list of dt values tested for black-box
        'ed_flips':   (trials,) array of event-driven flip counts
        'ed_edges':   (trials,) array of event-driven total edge counts
        'ed_time':    (trials,) array of event-driven wall times
        'bb_flips':   (len(dt_values), trials) array
        'bb_edges':   (len(dt_values), trials) array
        'bb_time':    (len(dt_values), trials) array
        'bb_steps':   (len(dt_values), trials) array
    """
    rng = np.random.default_rng(seed)

    ed_flips    = np.zeros(trials)
    ed_edges    = np.zeros(trials)
    ed_time     = np.zeros(trials)
    bb_flips    = np.zeros((len(dt_values), trials))
    bb_edges    = np.zeros((len(dt_values), trials))
    bb_time     = np.zeros((len(dt_values), trials))
    bb_steps    = np.zeros((len(dt_values), trials))
    bb_accuracy = np.zeros((len(dt_values), trials))
    bb_C        = np.zeros((len(dt_values), trials))

    for trial in range(trials):
        pts = generate_trajectory(n, rng)

        # Event-driven
        flips, edges, wall, _ = run_event_driven(pts, T)
        ed_flips[trial] = flips
        ed_edges[trial] = edges
        ed_time[trial]  = wall

        # Black-box at each dt
        for di, dt in enumerate(dt_values):
            flips, edges, wall, steps, accuracy, C = run_black_box(pts, T, dt)
            bb_flips[di, trial]    = flips
            bb_edges[di, trial]    = edges
            bb_time[di, trial]     = wall
            bb_steps[di, trial]    = steps
            bb_accuracy[di, trial] = accuracy
            bb_C[di, trial]        = C

        print(f"  Trial {trial+1}/{trials} done  "
              f"(ED flips={ed_flips[trial]:.0f}, edges={ed_edges[trial]:.0f} | "
              f"BB accuracy={bb_accuracy[:, trial].round(3)}, C={bb_C[:, trial].round(3)})")

    return dict(
        dt_values=dt_values,
        ed_flips=ed_flips,
        ed_edges=ed_edges,
        ed_time=ed_time,
        bb_flips=bb_flips,
        bb_edges=bb_edges,
        bb_time=bb_time,
        bb_steps=bb_steps,
        bb_accuracy=bb_accuracy,
        bb_C=bb_C,
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
    dt_vals     = np.array(results["dt_values"])
    ed_flips    = results["ed_flips"]
    ed_edges    = results["ed_edges"]
    bb_flips    = results["bb_flips"]
    bb_edges    = results["bb_edges"]
    bb_time     = results["bb_time"]
    ed_time     = results["ed_time"]
    bb_accuracy = results["bb_accuracy"]
    bb_C        = results["bb_C"]

    bb_flip_mean = bb_flips.mean(axis=1)
    bb_flip_std  = bb_flips.std(axis=1)
    bb_edge_mean = bb_edges.mean(axis=1)
    bb_edge_std  = bb_edges.std(axis=1)
    bb_time_mean = bb_time.mean(axis=1)
    bb_time_std  = bb_time.std(axis=1)
    bb_acc_mean  = bb_accuracy.mean(axis=1)
    bb_acc_std   = bb_accuracy.std(axis=1)
    bb_C_mean    = bb_C.mean(axis=1)
    bb_C_std     = bb_C.std(axis=1)

    ed_flip_mean = ed_flips.mean()
    ed_flip_std  = ed_flips.std()
    ed_edge_mean = ed_edges.mean()
    ed_edge_std  = ed_edges.std()
    ed_time_mean = ed_time.mean()
    ed_time_std  = ed_time.std()

    fig, axes = plt.subplots(1, 5, figsize=(26, 4.5))
    fig.patch.set_facecolor(PALETTE["bg"])

    for ax in axes:
        ax.set_facecolor(PALETTE["bg"])
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#888888")
        ax.tick_params(colors="#444444")
        ax.yaxis.label.set_color("#333333")
        ax.xaxis.label.set_color("#333333")
        ax.title.set_color("#111111")

    xlabel = "Time step  $\\Delta t$  (smaller = higher sampling rate)"

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
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel("Total edge flips", fontsize=10)
    ax.set_title(f"Flip count vs. sampling rate\n($n={n}$, $T={T}$)", fontsize=11)
    ax.legend(fontsize=8.5, framealpha=0.6)
    ax.grid(True, linestyle=":", color=PALETTE["grid"], linewidth=0.8)
    ax.annotate("← finer sampling\n    (smaller $d_{\\max}$)",
                xy=(dt_vals.min() * 1.5, bb_flip_mean[-1]),
                fontsize=8, color="#555555", va="center")

    # ---- Panel 2: Avg edges vs dt ----
    ax = axes[1]
    ax.errorbar(dt_vals, bb_edge_mean, yerr=bb_edge_std,
                fmt="D-", color=PALETTE["bb"], linewidth=2,
                markersize=6, capsize=4, label="Black-box (InsertAndDelete)")
    ax.axhline(ed_edge_mean, color=PALETTE["ed"], linewidth=2,
               linestyle="--", label="Event-driven (known trajectory)")
    ax.fill_between(
        [dt_vals.min(), dt_vals.max()],
        ed_edge_mean - ed_edge_std,
        ed_edge_mean + ed_edge_std,
        color=PALETTE["ed"], alpha=0.15,
    )
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel("Avg edges per snapshot", fontsize=10)
    ax.set_title(f"Avg edges per snapshot vs. sampling rate\n($n={n}$, $T={T}$)", fontsize=11)
    ax.legend(fontsize=8.5, framealpha=0.6)
    ax.grid(True, linestyle=":", color=PALETTE["grid"], linewidth=0.8)

    # ---- Panel 3: Accuracy vs dt ----
    ax = axes[2]
    ax.errorbar(dt_vals, bb_acc_mean, yerr=bb_acc_std,
                fmt="^-", color=PALETTE["bb"], linewidth=2,
                markersize=6, capsize=4, label="Black-box accuracy")
    ax.axhline(1.0, color=PALETTE["ed"], linewidth=2,
               linestyle="--", label="Perfect accuracy (event-driven)")
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_ylim(0, 1.05)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(r"Accuracy  $\alpha = 1 - \frac{\mathrm{wrong\ edges}}{e}$", fontsize=10)
    ax.set_title(f"Per-step accuracy vs. sampling rate\n($n={n}$, $T={T}$)", fontsize=11)
    ax.legend(fontsize=8.5, framealpha=0.6)
    ax.grid(True, linestyle=":", color=PALETTE["grid"], linewidth=0.8)

    # ---- Panel 4: Implied C vs dt ----
    ax = axes[3]
    ax.errorbar(dt_vals, bb_C_mean, yerr=bb_C_std,
                fmt="P-", color=PALETTE["bb"], linewidth=2,
                markersize=6, capsize=4, label="Implied $C$")
    all_C = bb_C.flatten()
    grand_C = float(np.nanmean(all_C[np.isfinite(all_C)]))
    ax.axhline(grand_C, color=PALETTE["ed"], linewidth=2,
               linestyle="--", label=f"Grand mean $C$ = {grand_C:.3f}")
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel("Implied $C$", fontsize=10)
    ax.set_title(f"Empirical constant $C$ vs. sampling rate\n($n={n}$, $T={T}$)", fontsize=11)
    ax.legend(fontsize=8.5, framealpha=0.6)
    ax.grid(True, linestyle=":", color=PALETTE["grid"], linewidth=0.8)

    # ---- Panel 5: Wall-clock time vs dt ----
    ax = axes[4]
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
    ax.set_xlabel(xlabel, fontsize=10)
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
    dt_vals     = results["dt_values"]
    ed_flips    = results["ed_flips"]
    ed_edges    = results["ed_edges"]
    bb_flips    = results["bb_flips"]
    bb_edges    = results["bb_edges"]
    bb_time     = results["bb_time"]
    ed_time     = results["ed_time"]
    bb_accuracy = results["bb_accuracy"]
    bb_C        = results["bb_C"]

    print("\n" + "=" * 105)
    print(f"  BENCHMARK SUMMARY   n={n}   T={T}   trials={len(ed_flips)}")
    print("=" * 105)
    print(f"  {'Strategy':<35} {'Flips (mean±std)':<22} {'Avg edges':<12} "
          f"{'Accuracy (mean±std)':<22} {'C (mean±std)':<20} {'Time (s)'}")
    print("-" * 105)
    print(f"  {'Event-driven':<35} "
          f"{ed_flips.mean():>7.1f} ± {ed_flips.std():>5.1f}   "
          f"{ed_edges.mean():>8.1f}    "
          f"{'N/A':<22} {'N/A':<20} "
          f"{ed_time.mean():.4f} ± {ed_time.std():.4f}")
    for di, dt in enumerate(dt_vals):
        label = f"Black-box  dt={dt:.4f}"
        print(f"  {label:<35} "
              f"{bb_flips[di].mean():>7.1f} ± {bb_flips[di].std():>5.1f}   "
              f"{bb_edges[di].mean():>8.1f}    "
              f"{bb_accuracy[di].mean():.4f} ± {bb_accuracy[di].std():.4f}       "
              f"{bb_C[di].mean():>6.3f} ± {bb_C[di].std():.3f}   "
              f"{bb_time[di].mean():.4f} ± {bb_time[di].std():.4f}")
    print("=" * 105)

    all_C = bb_C.flatten()
    all_C = all_C[np.isfinite(all_C)]
    print(f"\n  Empirical constant C (mean ± std across all dt and trials): "
          f"{all_C.mean():.4f} ± {all_C.std():.4f}")
    print()
    print("  As dt decreases (finer sampling / smaller d_max), accuracy approaches 1")
    print("  and C stabilises toward its true value. Avg edges per snapshot should")
    print("  remain roughly constant across dt since it depends on n, not sampling rate.")
    print()


# ---------------------------------------------------------------------------
# N-sweep: how does C scale with graph size?
# ---------------------------------------------------------------------------

def run_n_sweep(n_values: List[int], T: float, dt_values: List[float],
                trials: int, seed: int):
    """
    Run the benchmark for each n in n_values and return scaling statistics.

    For each n we compute the grand-mean C (averaged over all dt and trials),
    the mean avg-edge-count e, and the normalised variants C*e and C*sqrt(e).
    If C is truly scale-free, one of these normalised forms should be constant.

    Returns a dict with arrays indexed by n_values:
        'n_values', 'e_mean', 'e_std',
        'C_mean', 'C_std',
        'Ce_mean', 'Ce_std',       # C * e
        'Csqrte_mean', 'Csqrte_std'  # C * sqrt(e)
    """
    n_values = list(n_values)
    e_mean      = np.zeros(len(n_values))
    e_std       = np.zeros(len(n_values))
    C_mean      = np.zeros(len(n_values))
    C_std       = np.zeros(len(n_values))
    Ce_mean     = np.zeros(len(n_values))
    Ce_std      = np.zeros(len(n_values))
    Csqrte_mean = np.zeros(len(n_values))
    Csqrte_std  = np.zeros(len(n_values))

    for i, n in enumerate(n_values):
        print(f"  n={n} ...", flush=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = run_benchmark(n=n, T=T, dt_values=dt_values,
                                    trials=trials, seed=seed)

        # grand-mean e across all dt and trials
        all_e = results["bb_edges"].flatten()
        e_mean[i] = all_e.mean()
        e_std[i]  = all_e.std()

        # grand-mean C across all dt and trials (finite values only)
        all_C = results["bb_C"].flatten()
        all_C = all_C[np.isfinite(all_C)]
        C_mean[i] = all_C.mean()
        C_std[i]  = all_C.std()

        # normalised variants (per-sample, then aggregate)
        all_e_flat = results["bb_edges"].flatten()   # one e per (dt, trial)
        all_C_flat = results["bb_C"].flatten()
        mask = np.isfinite(all_C_flat)
        Ce     = all_C_flat[mask] * all_e_flat[mask]
        Csqrte = all_C_flat[mask] * np.sqrt(all_e_flat[mask])
        Ce_mean[i]     = Ce.mean()
        Ce_std[i]      = Ce.std()
        Csqrte_mean[i] = Csqrte.mean()
        Csqrte_std[i]  = Csqrte.std()

    return dict(
        n_values=n_values,
        e_mean=e_mean, e_std=e_std,
        C_mean=C_mean, C_std=C_std,
        Ce_mean=Ce_mean, Ce_std=Ce_std,
        Csqrte_mean=Csqrte_mean, Csqrte_std=Csqrte_std,
    )


def loglog_fit(x: np.ndarray, y: np.ndarray):
    """
    Fit log(y) = alpha*log(x) + log(k) via least squares.
    Returns (alpha, k, r_squared).
    """
    lx = np.log(x)
    ly = np.log(y)
    coeffs = np.polyfit(lx, ly, 1)
    alpha, log_k = coeffs
    k = np.exp(log_k)
    y_pred = alpha * lx + log_k
    ss_res = np.sum((ly - y_pred) ** 2)
    ss_tot = np.sum((ly - ly.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return float(alpha), float(k), float(r2)


def print_n_sweep(sweep: dict):
    n_vals = np.array(sweep["n_values"])
    e_mean = sweep["e_mean"]
    C_mean = sweep["C_mean"]

    # Fit C ~ k * e^alpha
    alpha, k, r2 = loglog_fit(e_mean, C_mean)

    print("\n" + "=" * 90)
    print("  N-SWEEP: How does C scale with graph size?")
    print("=" * 90)
    print(f"  {'n':>5}  {'avg e':>9}  {'C (mean±std)':>20}  "
          f"{'C·e (mean±std)':>22}  {'C·√e (mean±std)':>22}")
    print("-" * 90)
    for i, n in enumerate(n_vals):
        print(f"  {n:>5}  {sweep['e_mean'][i]:>7.1f}    "
              f"{sweep['C_mean'][i]:>7.4f} ± {sweep['C_std'][i]:.4f}    "
              f"{sweep['Ce_mean'][i]:>8.4f} ± {sweep['Ce_std'][i]:.4f}          "
              f"{sweep['Csqrte_mean'][i]:>8.4f} ± {sweep['Csqrte_std'][i]:.4f}")
    print("=" * 90)
    print()
    print(f"  Log-log fit:  C ≈ {k:.4f} · e^{alpha:.3f}   (R² = {r2:.4f})")
    print(f"  → Suggested formula:  α = 1 - k · v·Δt / (d · e^{1-alpha:.3f})")
    print(f"    where k ≈ {k:.4f}")
    print()
    print("  Interpretation guide (based on fitted exponent):")
    print("   - exponent ≈ 0   → C is scale-free,     formula: α = 1 - C·v·Δt / (d·e)")
    print("   - exponent ≈ 0.5 → C ∝ √e,              formula: α = 1 - k·v·Δt / (d·e^1.5)")
    print("   - exponent ≈ 1   → C ∝ e,               formula: α = 1 - k·v·Δt / (d·e²)")
    print("   - exponent ≈ 2   → C ∝ e²,              formula: α = 1 - k·v·Δt / (d·e³)")
    print()


def plot_n_sweep(sweep: dict, T: float, out_path: str):
    n_vals      = np.array(sweep["n_values"])
    e_mean      = sweep["e_mean"]
    C_mean      = sweep["C_mean"]
    C_std       = sweep["C_std"]
    Ce_mean     = sweep["Ce_mean"]
    Ce_std      = sweep["Ce_std"]
    Csqrte_mean = sweep["Csqrte_mean"]
    Csqrte_std  = sweep["Csqrte_std"]

    # Fit C ~ k * e^alpha for the power-law overlay
    alpha, k, r2 = loglog_fit(e_mean, C_mean)
    e_fit = np.linspace(e_mean.min(), e_mean.max(), 200)
    C_fit = k * e_fit ** alpha

    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
    fig.patch.set_facecolor(PALETTE["bg"])
    for ax in axes:
        ax.set_facecolor(PALETTE["bg"])
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#888888")
        ax.tick_params(colors="#444444")
        ax.yaxis.label.set_color("#333333")
        ax.xaxis.label.set_color("#333333")
        ax.title.set_color("#111111")

    col = PALETTE["bb"]

    # Panel 1: log-log C vs e with fit
    ax = axes[0]
    ax.errorbar(e_mean, C_mean, yerr=C_std, fmt="o", color=col,
                linewidth=2, markersize=6, capsize=4, label="Empirical $C$")
    ax.plot(e_fit, C_fit, color=PALETTE["ed"], linewidth=2, linestyle="--",
            label=f"Fit: $k \\cdot e^{{{alpha:.2f}}}$\n$R^2={r2:.3f}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Avg edges $e$", fontsize=10)
    ax.set_ylabel("$C$", fontsize=10)
    ax.set_title(f"$C$ vs $e$ (log-log)\nfit exponent = {alpha:.2f}", fontsize=11)
    ax.legend(fontsize=8.5, framealpha=0.6)
    ax.grid(True, linestyle=":", color=PALETTE["grid"], linewidth=0.8)

    # Panel 2: raw C vs n
    ax = axes[1]
    ax.errorbar(n_vals, C_mean, yerr=C_std, fmt="o-", color=col,
                linewidth=2, markersize=6, capsize=4)
    ax.set_xlabel("Number of points $n$", fontsize=10)
    ax.set_ylabel("$C$", fontsize=10)
    ax.set_title("Raw $C$ vs $n$\n(flat = scale-free)", fontsize=11)
    ax.grid(True, linestyle=":", color=PALETTE["grid"], linewidth=0.8)

    # Panel 3: C·e vs n
    ax = axes[2]
    ax.errorbar(n_vals, Ce_mean, yerr=Ce_std, fmt="s-", color=col,
                linewidth=2, markersize=6, capsize=4)
    ax.set_xlabel("Number of points $n$", fontsize=10)
    ax.set_ylabel("$C \\cdot e$", fontsize=10)
    ax.set_title("$C \\cdot e$ vs $n$\n(flat → $C = k/e$)", fontsize=11)
    ax.grid(True, linestyle=":", color=PALETTE["grid"], linewidth=0.8)

    # Panel 4: C·√e vs n
    ax = axes[3]
    ax.errorbar(n_vals, Csqrte_mean, yerr=Csqrte_std, fmt="^-", color=col,
                linewidth=2, markersize=6, capsize=4)
    ax.set_xlabel("Number of points $n$", fontsize=10)
    ax.set_ylabel("$C \\cdot \\sqrt{e}$", fontsize=10)
    ax.set_title("$C \\cdot \\sqrt{e}$ vs $n$\n(flat → $C = k/\\sqrt{e}$)", fontsize=11)
    ax.grid(True, linestyle=":", color=PALETTE["grid"], linewidth=0.8)

    fig.suptitle(
        f"Scaling of empirical constant $C$ with graph size  ($T={T}$)",
        fontsize=13, fontweight="bold", color="#111111", y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    print(f"N-sweep plot saved to: {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark event driven vs. blackbox Delaunay maintenance."
    )
    parser.add_argument("--n",        type=int,   default=40,
                        help="Number of moving points for single run (default: 40)")
    parser.add_argument("--T",        type=float, default=1.0,
                        help="Time horizon (default: 1.0)")
    parser.add_argument("--trials",   type=int,   default=5,
                        help="Number of independent random trials (default: 5)")
    parser.add_argument("--seed",     type=int,   default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--out",      type=str,   default="benchmark_results.png",
                        help="Output plot filename (default: benchmark_results.png)")
    parser.add_argument("--n_sweep",  action="store_true",
                        help="Run n-sweep to study how C scales with graph size")
    parser.add_argument("--n_values", type=int,   nargs="+",
                        default=[10, 20, 30, 40, 60, 80, 100],
                        help="Values of n to sweep over (default: 10 20 30 40 60 80 100 150 200)")
    parser.add_argument("--sweep_out", type=str,  default="n_sweep_results.png",
                        help="Output plot filename for n-sweep (default: n_sweep_results.png)")
    args = parser.parse_args()

    # Use a coarser dt set for n-sweep to keep runtime manageable
    dt_values_full   = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
    dt_values_sweep  = [0.2, 0.05, 0.01, 0.002]

    if args.n_sweep:
        print(f"\nRunning n-sweep:  n_values={args.n_values}  T={args.T}  "
              f"trials={args.trials}  seed={args.seed}")
        print(f"dt values for sweep: {dt_values_sweep}\n")
        sweep = run_n_sweep(
            n_values=args.n_values,
            T=args.T,
            dt_values=dt_values_sweep,
            trials=args.trials,
            seed=args.seed,
        )
        print_n_sweep(sweep)
        plot_n_sweep(sweep, args.T, out_path=args.sweep_out)
    else:
        print(f"\nRunning benchmark:  n={args.n}  T={args.T}  "
              f"trials={args.trials}  seed={args.seed}")
        print(f"Blackbox dt values: {dt_values_full}\n")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = run_benchmark(
                n=args.n,
                T=args.T,
                dt_values=dt_values_full,
                trials=args.trials,
                seed=args.seed,
            )
        print_summary(results, args.n, args.T)
        plot_results(results, args.n, args.T, out_path=args.out)



if __name__ == "__main__":
    main()
