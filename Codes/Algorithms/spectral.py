"""Spectral projection experiments for contaminated max-affine data.

Usage:
    .\\mainenv\\Scripts\\python.exe Codes\\Algorithms\\spectral.py --nstart 100 --nend 1000 --njump 100

Example:
    .\\mainenv\\Scripts\\python.exe Codes\\Algorithms\\spectral.py --nstart 100 --nend 1000 --njump 100 --m 20 --d 5 --k 2 --alpha 0.1 --M 5 --seed 1

Arguments:
    --nstart:
        First sample size in the sweep.
        Default: 100

    --nend:
        Last sample size in the sweep, included when it falls on the njump grid.
        Default: 1000

    --njump:
        Step size between sample sizes.
        Default: 100

    --m:
        Number of Monte Carlo runs for each sample size.
        Default: 10

    --d:
        Covariate dimension. Each covariate x_i is sampled from N(0, I_d).
        Default: 5

    --k:
        Number of max-affine pieces, equivalently the number of beta vectors.
        Default: 2

    --alpha:
        Fraction of response values to replace with outliers.
        Default: 0.1

    --M:
        Outlier bound. Contaminated response values are sampled from Unif[-M, M].
        Default: 5.0

    --seed:
        Optional random seed for reproducible experiments.
        Default: None
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Codes.utilities import sample_max_affine_vectors


RESULTS_DIR = PROJECT_ROOT / "results" / "spectral"


def generate_clean_dataset(
    beta: np.ndarray,
    n: int,
    rng: Any,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate clean max-affine samples."""
    d = beta.shape[1] - 1
    x = rng.standard_normal(size=(n, d))
    augmented_x = np.column_stack((x, np.ones(n)))
    y = np.max(augmented_x @ beta.T, axis=1)
    return x, y


def contaminate_responses(
    y: np.ndarray,
    alpha: float,
    outlier_bound: float,
    rng: Any,
) -> Tuple[np.ndarray, np.ndarray]:
    """Replace alpha fraction of responses with uniform outliers."""
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1.")
    if outlier_bound <= 0:
        raise ValueError("M must be positive.")

    contaminated_y = y.copy()
    n_outliers = int(round(alpha * len(y)))
    outlier_indices = rng.choice(len(y), size=n_outliers, replace=False)
    contaminated_y[outlier_indices] = rng.uniform(
        low=-outlier_bound,
        high=outlier_bound,
        size=n_outliers,
    )
    return contaminated_y, outlier_indices


def projection_onto_span(vectors: np.ndarray, rank: Optional[int] = None) -> np.ndarray:
    """Return the orthogonal projection onto the row span of vectors."""
    _, _, vh = np.linalg.svd(vectors, full_matrices=False)
    subspace_dim = min(rank or vectors.shape[0], vh.shape[0])
    basis = vh[:subspace_dim].T
    return basis @ basis.T


def spectral_projection(x: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
    """Estimate the projection matrix from one subdataset."""
    n_sub, d = x.shape
    identity = np.eye(d)

    m1 = np.mean(y[:, None] * x, axis=0)
    centered_second_moments = x[:, :, None] * x[:, None, :] - identity
    m2 = np.mean(y[:, None, None] * centered_second_moments, axis=0)
    moment_matrix = np.outer(m1, m1) + m2
    moment_matrix = 0.5 * (moment_matrix + moment_matrix.T)

    return top_eigen_projection(moment_matrix, k)


def top_eigen_projection(moment_matrix: np.ndarray, k: int) -> np.ndarray:
    """Return the projection onto the span of the k largest eigenvalue directions."""
    symmetric_matrix = 0.5 * (moment_matrix + moment_matrix.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric_matrix)
    top_indices = np.argsort(eigenvalues)[-k:]
    u = eigenvectors[:, top_indices]
    return u @ u.T


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run spectral projection experiments for contaminated max-affine data."
    )
    parser.add_argument("--nstart", type=int, default=100, help="First sample size.")
    parser.add_argument("--nend", type=int, default=1000, help="Last sample size.")
    parser.add_argument("--njump", type=int, default=100, help="Sample size step.")
    parser.add_argument("--m", type=int, default=10, help="Monte Carlo runs per sample size.")
    parser.add_argument("--d", type=int, default=5, help="Covariate dimension.")
    parser.add_argument("--k", type=int, default=2, help="Number of max-affine pieces.")
    parser.add_argument("--alpha", type=float, default=0.1, help="Outlier fraction.")
    parser.add_argument("--M", type=float, default=5.0, help="Uniform outlier bound.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Directory where the experiment data file is saved.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.nstart <= 0:
        raise ValueError("nstart must be positive.")
    if args.nend < args.nstart:
        raise ValueError("nend must be greater than or equal to nstart.")
    if args.njump <= 0:
        raise ValueError("njump must be positive.")
    if args.m <= 0:
        raise ValueError("m must be positive.")
    if args.d <= 0:
        raise ValueError("d must be positive.")
    if args.k <= 0:
        raise ValueError("k must be positive.")
    if args.k > args.d:
        raise ValueError("k must be no larger than d for a k-dimensional subspace in R^d.")
    if not 0 <= args.alpha < 1:
        raise ValueError("alpha must be at least 0 and less than 1.")
    if args.M <= 0:
        raise ValueError("M must be positive.")


def sample_sizes(args: argparse.Namespace) -> np.ndarray:
    return np.arange(args.nstart, args.nend + 1, args.njump, dtype=int)


def run_trial(n: int, args: argparse.Namespace, rng: Any) -> Tuple[float, float, int]:
    beta = sample_max_affine_vectors(args.k, args.d, rng=rng)
    x, clean_y = generate_clean_dataset(beta, n, rng)
    contaminated_y, outlier_indices = contaminate_responses(clean_y, args.alpha, args.M, rng)
    clean_mask = np.ones(n, dtype=bool)
    clean_mask[outlier_indices] = False

    pihat_total = spectral_projection(x, contaminated_y, args.k)
    pihat_clean = spectral_projection(x[clean_mask], clean_y[clean_mask], args.k)
    groundtruth_pi = projection_onto_span(beta[:, : args.d], rank=args.k)

    total_distance = np.linalg.norm(pihat_total - groundtruth_pi, ord="fro")
    clean_distance = np.linalg.norm(pihat_clean - groundtruth_pi, ord="fro")
    return total_distance, clean_distance, len(outlier_indices)


def format_value(value: Any) -> str:
    return str(value).replace(".", "p").replace("-", "m")


def result_filename(args: argparse.Namespace) -> str:
    seed_value = "none" if args.seed is None else args.seed
    parts = [
        "spectral",
        f"k{args.k}",
        f"d{args.d}",
        f"alpha{format_value(args.alpha)}",
        f"M{format_value(args.M)}",
        f"n{args.nstart}-{args.nend}-{args.njump}",
        f"m{args.m}",
        f"seed{seed_value}",
    ]
    return "_".join(parts) + ".npz"


def save_results(
    args: argparse.Namespace,
    n_values: np.ndarray,
    total_distances: np.ndarray,
    clean_distances: np.ndarray,
    outlier_counts: np.ndarray,
) -> Path:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / result_filename(args)
    metadata = {
        "nstart": args.nstart,
        "nend": args.nend,
        "njump": args.njump,
        "m": args.m,
        "d": args.d,
        "k": args.k,
        "alpha": args.alpha,
        "M": args.M,
        "seed": args.seed,
    }
    np.savez(
        str(output_path),
        n_values=n_values,
        total_distances=total_distances,
        clean_distances=clean_distances,
        outlier_counts=outlier_counts,
        metadata=json.dumps(metadata, sort_keys=True),
    )
    return output_path


def print_progress(n_index: int, n_total: int, n: int, run_index: int, m: int) -> None:
    message = (
        f"Progress: n {n_index + 1}/{n_total} "
        f"(n={n}), Monte Carlo {run_index + 1}/{m}"
    )
    print("\r\033[K" + message, end="", flush=True)


def main() -> None:
    args = parse_args()
    validate_args(args)

    rng = np.random.default_rng(args.seed)
    n_values = sample_sizes(args)
    total_distances = np.zeros((len(n_values), args.m))
    clean_distances = np.zeros((len(n_values), args.m))
    outlier_counts = np.zeros((len(n_values), args.m), dtype=int)

    for n_index, n in enumerate(n_values):
        for run_index in range(args.m):
            print_progress(n_index, len(n_values), int(n), run_index, args.m)
            total_distance, clean_distance, n_outliers = run_trial(int(n), args, rng)
            total_distances[n_index, run_index] = total_distance
            clean_distances[n_index, run_index] = clean_distance
            outlier_counts[n_index, run_index] = n_outliers

    output_path = save_results(args, n_values, total_distances, clean_distances, outlier_counts)
    print("\r\033[K", end="")
    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
