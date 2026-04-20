# CS506 Final Project

## Setup

Use **Python 3** (3.9+ recommended).

Install third-party dependencies from the project directory:

```bash
pip install numpy matplotlib
```

Or with `pip3`:

```bash
pip3 install numpy matplotlib
```

These packages are required by:

| Package       | Used for                                      |
| ------------- | --------------------------------------------- |
| **numpy**     | Random point generation (`test_points.py`) and arrays in the convex hull demo |
| **matplotlib**| Plotting the point set and hull (`graph_points.py`) |

Everything else comes from the Python standard library.

## Running the convex hull demo

From the repository root:

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
