"""Sweep total-L1 nospectral initialization over an (n, d) grid.

This script fixes the high-level configuration to the best options recorded in
``best options.md`` and varies only sample size n and covariate dimension d.
For each (n, d, Monte Carlo run), it generates one problem, scales random
initializations, screens them with a small number of RAM steps, and applies
the full Robust-AM iteration budget to the selected initialization.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Codes.Algorithms.random_search import (  # noqa: E402
    full_dimensional_random_search,
    lad_regression_admm,
    permutation_matched_normalized_squared_error,
)
from Codes.Algorithms.spectral import (  # noqa: E402
    contaminate_responses,
    format_value,
    generate_clean_dataset,
)
from Codes.utilities import sample_max_affine_vectors  # noqa: E402


RESULTS_DIR = PROJECT_ROOT / "results" / "sweeper"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep total-L1 nospectral initialization over n,d pairs."
    )
    parser.add_argument("--nstart", type=int, default=100, help="First sample size.")
    parser.add_argument("--nend", type=int, default=2500, help="Last sample size.")
    parser.add_argument("--njump", type=int, default=200, help="Sample size step.")
    parser.add_argument("--dstart", type=int, default=20, help="First dimension.")
    parser.add_argument("--dend", type=int, default=100, help="Last dimension.")
    parser.add_argument("--djump", type=int, default=10, help="Dimension step.")
    parser.add_argument("--m", type=int, default=20, help="Monte Carlo runs per grid cell.")
    parser.add_argument("--k", type=int, default=4, help="Number of max-affine pieces.")
    parser.add_argument("--alpha", type=float, default=0.2, help="Outlier fraction.")
    parser.add_argument("--M", type=float, default=50.0, help="Uniform outlier bound.")
    parser.add_argument(
        "--Mrand",
        type=int,
        default=20,
        help="Number of random-search initializations per problem.",
    )
    parser.add_argument(
        "--refinement-steps",
        type=int,
        default=5,
        help="Robust-AM refinement steps applied to every sampled candidate.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=50,
        help="Maximum additional Robust-AM refinements for the selected candidate.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1e-3,
        help="Normalized-error threshold used to mark final recovery success.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(28, max(1, os.cpu_count() or 1)),
        help="Parallel worker processes for candidate refinement.",
    )
    parser.add_argument("--seed", type=int, default=1, help="Random seed.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Directory where the sweep data file is saved.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.nstart <= 0:
        raise ValueError("nstart must be positive.")
    if args.nend < args.nstart:
        raise ValueError("nend must be greater than or equal to nstart.")
    if args.njump <= 0:
        raise ValueError("njump must be positive.")
    if args.dstart <= 0:
        raise ValueError("dstart must be positive.")
    if args.dend < args.dstart:
        raise ValueError("dend must be greater than or equal to dstart.")
    if args.djump <= 0:
        raise ValueError("djump must be positive.")
    if args.m <= 0:
        raise ValueError("m must be positive.")
    if args.k <= 0:
        raise ValueError("k must be positive.")
    if args.k > args.dstart:
        raise ValueError("k must be no larger than the smallest d.")
    if not 0 <= args.alpha < 1:
        raise ValueError("alpha must be at least 0 and less than 1.")
    if args.M <= 0:
        raise ValueError("M must be positive.")
    if args.Mrand <= 0:
        raise ValueError("Mrand must be positive.")
    if args.refinement_steps < 0:
        raise ValueError("refinement-steps must be nonnegative.")
    if args.max_iterations <= 0:
        raise ValueError("max-iterations must be positive.")
    if args.threshold <= 0:
        raise ValueError("threshold must be positive.")
    if args.workers <= 0:
        raise ValueError("workers must be positive.")


def grid_values(args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray]:
    n_values = np.arange(args.nstart, args.nend + 1, args.njump, dtype=int)
    d_values = np.arange(args.dstart, args.dend + 1, args.djump, dtype=int)
    return n_values, d_values


def result_filename(args: argparse.Namespace) -> str:
    seed_value = "none" if args.seed is None else args.seed
    parts = [
        "sweeper",
        "total_l1",
        "nospectral",
        f"k{args.k}",
        f"d{args.dstart}-{args.dend}-{args.djump}",
        f"n{args.nstart}-{args.nend}-{args.njump}",
        f"alpha{format_value(args.alpha)}",
        f"M{format_value(args.M)}",
        f"Mrand{args.Mrand}",
        "scalefirst",
        f"ref{args.refinement_steps}",
        f"finaliter{args.max_iterations}",
        f"threshold{format_value(args.threshold)}",
        f"m{args.m}",
        f"seed{seed_value}",
    ]
    return "_".join(parts) + ".npz"


def refine_selected_candidate(
    x: np.ndarray,
    y: np.ndarray,
    beta_initial: np.ndarray,
    beta_truth: np.ndarray,
    max_iterations: int,
    threshold: float,
) -> Tuple[np.ndarray, int, bool]:
    """Refine one selected candidate on a fixed problem."""
    x_aug = np.column_stack((x, np.ones(x.shape[0])))
    beta = beta_initial.copy()
    previous_assignments = None

    for iteration in range(max_iterations):
        assignments = np.argmax(x_aug @ beta.T, axis=1)
        if previous_assignments is not None and np.array_equal(
            assignments, previous_assignments
        ):
            error = permutation_matched_normalized_squared_error(beta, beta_truth)
            return beta, iteration, error < threshold
        previous_assignments = assignments.copy()

        beta_next = beta.copy()
        for component in range(beta.shape[0]):
            mask = assignments == component
            if np.any(mask):
                beta_next[component] = lad_regression_admm(
                    x_aug[mask], y[mask], beta_init=beta[component]
                )
        beta = beta_next
    error = permutation_matched_normalized_squared_error(beta, beta_truth)
    return beta, max_iterations, error < threshold


def run_cell(
    n: int,
    d: int,
    args: argparse.Namespace,
    rng: Any,
) -> Tuple[float, float, int, bool, np.ndarray]:
    beta = sample_max_affine_vectors(args.k, d, rng=rng)
    x, clean_y = generate_clean_dataset(beta, n, rng)
    contaminated_y, _ = contaminate_responses(clean_y, args.alpha, args.M, rng)
    beta_total_l1, info_total_l1 = full_dimensional_random_search(
        x,
        contaminated_y,
        args.k,
        args.Mrand,
        random_state=rng,
        loss="l1",
        refinement_steps=args.refinement_steps,
        workers=args.workers,
    )
    c_value = float(info_total_l1["best_c"])
    final_beta, iterations, stopped_early = refine_selected_candidate(
        x,
        contaminated_y,
        beta_total_l1,
        beta,
        args.max_iterations,
        args.threshold,
    )
    error = permutation_matched_normalized_squared_error(final_beta, beta)
    return float(error), c_value, iterations, stopped_early, final_beta


def print_progress(
    d_index: int,
    d_total: int,
    d: int,
    n_index: int,
    n_total: int,
    n: int,
    run_index: int,
    m: int,
) -> None:
    message = (
        f"Progress: d {d_index + 1}/{d_total} (d={d}), "
        f"n {n_index + 1}/{n_total} (n={n}), Monte Carlo {run_index + 1}/{m}"
    )
    print("\r\033[K" + message, end="", flush=True)


def save_results(
    args: argparse.Namespace,
    n_values: np.ndarray,
    d_values: np.ndarray,
    best_errors: np.ndarray,
    best_c_values: np.ndarray,
    iterations_used: np.ndarray,
    stopped_early: np.ndarray,
    best_betas: np.ndarray,
    best_beta_masks: np.ndarray,
) -> Path:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / result_filename(args)
    metadata: Dict[str, Any] = {
        "mode": "nospectral",
        "selection": "total_l1",
        "loss": "l1",
        "nstart": args.nstart,
        "nend": args.nend,
        "njump": args.njump,
        "dstart": args.dstart,
        "dend": args.dend,
        "djump": args.djump,
        "m": args.m,
        "k": args.k,
        "alpha": args.alpha,
        "M": args.M,
        "Mrand": args.Mrand,
        "refinement_steps": args.refinement_steps,
        "max_iterations": args.max_iterations,
        "threshold": args.threshold,
        "workers": args.workers,
        "seed": args.seed,
        "metric": "permutation_matched_normalized_squared_error",
        "protocol": "fixed_problem_select_then_refine",
        "defaults_source": "best options.md",
    }
    np.savez(
        str(output_path),
        n_values=n_values,
        d_values=d_values,
        best_errors=best_errors,
        best_c_values=best_c_values,
        iterations_used=iterations_used,
        stopped_early=stopped_early,
        best_betas=best_betas,
        best_beta_masks=best_beta_masks,
        metadata=json.dumps(metadata, sort_keys=True),
    )
    return output_path


def load_existing_results(
    args: argparse.Namespace,
    shape: Tuple[int, int, int],
    max_beta_width: int,
) -> Optional[Dict[str, np.ndarray]]:
    output_path = args.output_dir / result_filename(args)
    if not output_path.exists():
        return None

    data = np.load(str(output_path), allow_pickle=False)
    expected_best_beta_shape = shape + (args.k, max_beta_width)
    required_shapes = {
        "best_errors": shape,
        "best_c_values": shape,
        "iterations_used": shape,
        "stopped_early": shape,
        "best_betas": expected_best_beta_shape,
        "best_beta_masks": expected_best_beta_shape,
    }
    for key, expected_shape in required_shapes.items():
        if key not in data or data[key].shape != expected_shape:
            return None

    return {key: data[key].copy() for key in required_shapes}


def main() -> None:
    args = parse_args()
    validate_args(args)

    rng = np.random.default_rng(args.seed)
    n_values, d_values = grid_values(args)
    shape = (len(d_values), len(n_values), args.m)
    best_errors = np.zeros(shape)
    best_c_values = np.zeros(shape)
    iterations_used = np.zeros(shape, dtype=int)
    stopped_early = np.zeros(shape, dtype=bool)
    max_beta_width = int(d_values[-1]) + 1
    best_betas = np.zeros(shape + (args.k, max_beta_width))
    best_beta_masks = np.zeros(shape + (args.k, max_beta_width), dtype=bool)

    existing = load_existing_results(args, shape, max_beta_width)
    if existing is not None:
        best_errors = existing["best_errors"]
        best_c_values = existing["best_c_values"]
        iterations_used = existing["iterations_used"]
        stopped_early = existing["stopped_early"]
        best_betas = existing["best_betas"]
        best_beta_masks = existing["best_beta_masks"]

    for d_index, d in enumerate(d_values):
        for n_index, n in enumerate(n_values):
            for run_index in range(args.m):
                if iterations_used[d_index, n_index, run_index] > 0:
                    continue
                print_progress(
                    d_index,
                    len(d_values),
                    int(d),
                    n_index,
                    len(n_values),
                    int(n),
                    run_index,
                    args.m,
                )
                error, c_value, attempts, hit_threshold, beta = run_cell(
                    int(n),
                    int(d),
                    args,
                    rng,
                )
                best_errors[d_index, n_index, run_index] = error
                best_c_values[d_index, n_index, run_index] = c_value
                iterations_used[d_index, n_index, run_index] = attempts
                stopped_early[d_index, n_index, run_index] = hit_threshold
                beta_width = beta.shape[1]
                best_betas[d_index, n_index, run_index, :, :beta_width] = beta
                best_beta_masks[d_index, n_index, run_index, :, :beta_width] = True
                save_results(
                    args,
                    n_values,
                    d_values,
                    best_errors,
                    best_c_values,
                    iterations_used,
                    stopped_early,
                    best_betas,
                    best_beta_masks,
                )

    output_path = save_results(
        args,
        n_values,
        d_values,
        best_errors,
        best_c_values,
        iterations_used,
        stopped_early,
        best_betas,
        best_beta_masks,
    )
    print("\r\033[K", end="")
    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
