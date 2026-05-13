# CS506 Final Project

## Setup

Use **Python 3** (3.9+ recommended).

### Virtual environment

From the repository root, create and activate a virtual environment, then install dependencies from `requirements.txt`.

**Linux / macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Windows (PowerShell)**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Windows (cmd.exe)**

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
pip install -r requirements.txt
```

Third-party packages are listed in `requirements.txt`:

| Package        | Used for |
| -------------- | -------- |
| **numpy**      | Arrays and random point generation (`test_points.py`, `graph_points.py`, `convex_hull_simple.py`, `voronoi_benchmark.py`) |
| **matplotlib** | Plotting (`graph_points.py`, `convex_hull_simple.py`, `voronoi_benchmark.py`) |
| **scipy**      | `scipy.spatial.Delaunay` in `voronoi_benchmark.py` |

Everything else comes from the Python standard library.

## Running the Voronoi benchmark

The script `voronoi_benchmark.py` compares event-driven versus black-box Delaunay maintenance for moving points and writes a figure to disk (non-interactive `Agg` backend).

With the venv activated, from the repository root:

```bash
python voronoi_benchmark.py --n 60 --T 1.0 --trials 10 --seed 42 --out fig_benchmark.png
```

Use `python3` instead of `python` if that is how Python is installed on your system.

Common flags include `--n` (number of points), `--T` (time horizon), `--trials` (random repeats), `--seed`, and `--out` (output image path). Run `python voronoi_benchmark.py --help` for the full argument list.

## Running the convex hull demo

From the repository root (with the venv activated):

```bash
python3 convex_hull_simple.py
```

On Windows, if `python3` is not on your `PATH`:

```bash
python convex_hull_simple.py
```

This generates random points, builds the hull, opens a figure, and shows the convex hull overlay. Close the plot window to exit.

To run without a GUI (e.g. headless CI), set a non-interactive backend before running; the script may still warn on `plt.show()`, but processing will complete:

```bash
# Linux / macOS
MPLBACKEND=Agg python3 convex_hull_simple.py
```

```powershell
# Windows PowerShell
$env:MPLBACKEND='Agg'; python convex_hull_simple.py
```
