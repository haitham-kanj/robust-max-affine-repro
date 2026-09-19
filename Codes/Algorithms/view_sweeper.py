"""Plot sweeper results as a capped grayscale heatmap.

Usage:
    python Codes/Algorithms/view_sweeper.py --data results/sweeper/sweeper_...npz
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot capped normalized errors from the n,d sweeper."
    )
    parser.add_argument("--data", type=Path, required=True, help="Saved sweeper .npz file.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path. Defaults to the data filename with .png extension.",
    )
    parser.add_argument(
        "--statistic",
        choices=("median", "mean", "min"),
        default="median",
        help="Monte Carlo statistic to show for each n,d cell.",
    )
    parser.add_argument(
        "--cap",
        type=float,
        default=1.0,
        help="Maximum displayed normalized error; larger values are clipped.",
    )
    parser.add_argument("--d-min", type=float, default=None)
    parser.add_argument("--d-max", type=float, default=None)
    return parser.parse_args()


def load_results(data_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    data = np.load(str(data_path))
    metadata = json.loads(str(data["metadata"]))
    return data["n_values"], data["d_values"], data["best_errors"], metadata


def configure_matplotlib() -> Any:
    os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib-cache"))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "mathtext.rm": "serif",
            "font.size": 8,
            "axes.labelsize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.9,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 4,
            "ytick.major.size": 4,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
        }
    )
    return plt


def aggregate_errors(best_errors: np.ndarray, statistic: str) -> np.ndarray:
    if statistic == "median":
        return np.median(best_errors, axis=2)
    if statistic == "mean":
        return np.mean(best_errors, axis=2)
    if statistic == "min":
        return np.min(best_errors, axis=2)
    raise ValueError(f"Unknown statistic: {statistic}")


def cell_edges(values: np.ndarray) -> np.ndarray:
    if values.ndim != 1 or values.size == 0:
        raise ValueError("Grid values must be a nonempty one-dimensional array.")
    if values.size == 1:
        return np.array([values[0] - 0.5, values[0] + 0.5], dtype=float)
    midpoints = 0.5 * (values[:-1] + values[1:])
    first = values[0] - 0.5 * (values[1] - values[0])
    last = values[-1] + 0.5 * (values[-1] - values[-2])
    return np.concatenate(([first], midpoints, [last])).astype(float)


def format_metadata_value(value: Any) -> Any:
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def plot_results(
    n_values: np.ndarray,
    d_values: np.ndarray,
    best_errors: np.ndarray,
    metadata: Dict[str, Any],
    output_path: Path,
    statistic: str,
    cap: float,
    d_min: float = None,
    d_max: float = None,
) -> None:
    plt = configure_matplotlib()
    capped = np.clip(aggregate_errors(best_errors, statistic), 0.0, cap)
    fig, ax = plt.subplots(figsize=(3.35, 2.55))
    d_edges = cell_edges(d_values)
    n_edges = cell_edges(n_values)
    lower_d = float(d_edges[0]) if d_min is None else d_min
    upper_d = float(d_edges[-1]) if d_max is None else d_max
    visible_d = d_values[(d_values >= lower_d) & (d_values <= upper_d)]

    mesh = ax.pcolormesh(
        d_edges,
        n_edges,
        capped.T,
        cmap="Greys",
        vmin=0.0,
        vmax=cap,
        shading="flat",
        edgecolors="face",
    )
    ax.set_xlabel(r"$d$")
    ax.set_ylabel(r"$n$")
    ax.set_xlim(lower_d, upper_d)
    ax.set_ylim(float(n_edges[0]), float(n_edges[-1]))
    ax.set_xticks(visible_d)
    ax.set_yticks(n_values[::2])
    ax.tick_params(top=False, right=False, pad=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    from mpl_toolkits.axes_grid1 import make_axes_locatable

    divider = make_axes_locatable(ax)
    colorbar_axis = divider.append_axes("right", size="4%", pad=0.08)
    colorbar = fig.colorbar(mesh, cax=colorbar_axis)
    colorbar.set_ticks(np.linspace(0.0, cap, 5))
    colorbar.ax.tick_params(labelsize=7, length=2, width=0.6, pad=2)

    fig.subplots_adjust(left=0.15, right=0.88, bottom=0.17, top=0.98)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_path = args.output if args.output is not None else args.data.with_suffix(".png")
    n_values, d_values, best_errors, metadata = load_results(args.data)
    plot_results(
        n_values,
        d_values,
        best_errors,
        metadata,
        output_path,
        args.statistic,
        args.cap,
        args.d_min,
        args.d_max,
    )
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
