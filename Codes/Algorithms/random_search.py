"""Random-search initialization experiment, with optional spectral subspace.

Paper-to-NumPy mapping:
    In spectral mode, xi_i is X[i] with shape (d,), y_i is y[i], and U_hat is a (d, k)
    NumPy array with orthonormal columns. The augmented subspace V_hat is never
    formed explicitly; instead, for augmented covariates [xi_i; 1],
    <[xi_i; 1], V_hat nu_j> equals <[X[i] @ U_hat; 1], nu_j>. Thus the code
    works in the reduced coordinates Z = [X @ U_hat, ones], shape (n, k + 1).

    Random vectors nu have shape (M, k, k + 1), where nu[ell, j] is
    nu_j^ell. Predictions have shape (M, n), where predictions[ell, i] is
    max_j <[X[i] @ U_hat; 1], nu[ell, j]>.

    In nospectral mode, candidates beta[ell, j] are sampled directly and
    uniformly from the unit ball in R^(d+1), scaled, refined, and then selected.
"""

import argparse
import itertools
import json
import multiprocessing as mp
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

from Codes.Algorithms.spectral import (  # noqa: E402
    contaminate_responses,
    format_value,
    generate_clean_dataset,
)
from Codes.utilities import sample_max_affine_vectors  # noqa: E402


RESULTS_DIR = PROJECT_ROOT / "results" / "random_search"
_REFINE_X_AUG = None
_REFINE_Y = None
_REFINE_STEPS = 0


def spectral_moment_matrix(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return the symmetric spectral moment matrix used by the first step."""
    _, d = x.shape
    identity = np.eye(d)
    m1 = np.mean(y[:, None] * x, axis=0)
    centered_second_moments = x[:, :, None] * x[:, None, :] - identity
    m2 = np.mean(y[:, None, None] * centered_second_moments, axis=0)
    moment_matrix = np.outer(m1, m1) + m2
    return 0.5 * (moment_matrix + moment_matrix.T)


def spectral_subspace(x: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
    """Return U_hat, the top-k eigenspace basis from the spectral moment."""
    moment_matrix = spectral_moment_matrix(x, y)
    eigenvalues, eigenvectors = np.linalg.eigh(moment_matrix)
    top_indices = np.argsort(eigenvalues)[-k:]
    return eigenvectors[:, top_indices]


def validate_random_search_inputs(
    X: np.ndarray,
    y: np.ndarray,
    U_hat: np.ndarray,
    M: int,
    check_orthonormal: bool,
) -> Tuple[int, int, int]:
    if X.ndim != 2:
        raise ValueError("X must have shape (n, d).")
    if y.ndim != 1:
        raise ValueError("y must have shape (n,).")
    n, d = X.shape
    if y.shape[0] != n:
        raise ValueError("X and y must have the same number of samples.")
    if U_hat.ndim != 2:
        raise ValueError("U_hat must have shape (d, k).")
    if U_hat.shape[0] != d:
        raise ValueError("U_hat must have the same row dimension as X.")
    k = U_hat.shape[1]
    if k <= 0:
        raise ValueError("U_hat must have at least one column.")
    if M <= 0:
        raise ValueError("M must be positive.")
    if n < 2:
        raise ValueError("At least two samples are needed for the selection split.")
    if check_orthonormal:
        gram = U_hat.T @ U_hat
        if not np.allclose(gram, np.eye(k), atol=1e-6):
            raise ValueError("U_hat columns are not approximately orthonormal.")
    return n, d, k


def sample_uniform_ball(
    rng: Any,
    size: Tuple[int, ...],
    dimension: int,
) -> np.ndarray:
    """Sample vectors uniformly from the Euclidean unit ball in R^dimension."""
    directions = rng.standard_normal(size=size + (dimension,))
    norms = np.linalg.norm(directions, axis=-1, keepdims=True)
    while np.any(norms == 0):
        zero_mask = (norms[..., 0] == 0)
        directions[zero_mask] = rng.standard_normal(size=(np.sum(zero_mask), dimension))
        norms = np.linalg.norm(directions, axis=-1, keepdims=True)

    directions = directions / norms
    radii = rng.random(size=size + (1,)) ** (1.0 / dimension)
    return directions * radii


def optimal_c_l1(y: np.ndarray, p: np.ndarray, eps: float = 1e-12) -> float:
    """Solve min_{c >= 0} sum_i |y_i - c p_i| by weighted median.

    For nonzero p_i, |y_i - c p_i| = |p_i| |c - y_i / p_i|, so the
    unconstrained L1 minimizer is a weighted median of ratios y_i / p_i with
    weights |p_i|. The nonnegative constraint projects that median to c >= 0.
    """
    if y.ndim != 1 or p.ndim != 1:
        raise ValueError("y and p must be one-dimensional.")
    if y.shape[0] != p.shape[0]:
        raise ValueError("y and p must have the same length.")

    mask = np.abs(p) > eps
    if not np.any(mask):
        return 0.0

    ratios = y[mask] / p[mask]
    weights = np.abs(p[mask])
    order = np.argsort(ratios)
    sorted_ratios = ratios[order]
    sorted_weights = weights[order]

    cumulative_weights = np.cumsum(sorted_weights)
    half_weight = 0.5 * sorted_weights.sum()
    index = int(np.searchsorted(cumulative_weights, half_weight, side="left"))
    index = min(index, sorted_ratios.shape[0] - 1)
    return float(max(0.0, sorted_ratios[index]))


def optimal_c_l2(y: np.ndarray, p: np.ndarray) -> float:
    """Solve min_{c >= 0} sum_i (y_i - c p_i)^2 in closed form."""
    denominator = float(np.dot(p, p))
    if denominator <= np.finfo(float).eps:
        return 0.0
    return float(max(0.0, np.dot(y, p) / denominator))


def soft_threshold(values: np.ndarray, threshold: float) -> np.ndarray:
    """Elementwise soft thresholding."""
    return np.sign(values) * np.maximum(np.abs(values) - threshold, 0.0)


def safe_lstsq(
    A: np.ndarray,
    b: np.ndarray,
    fallback: Optional[np.ndarray] = None,
    ridge: float = 1e-8,
) -> np.ndarray:
    """Return a least-squares solution with a ridge fallback for bad SVD cases."""
    if not np.all(np.isfinite(A)) or not np.all(np.isfinite(b)):
        if fallback is not None:
            return fallback.copy()
        raise np.linalg.LinAlgError("least-squares inputs contain nonfinite values")

    try:
        return np.linalg.lstsq(A, b, rcond=None)[0]
    except np.linalg.LinAlgError:
        normal_lhs = A.T @ A
        normal_rhs = A.T @ b
        if not np.all(np.isfinite(normal_lhs)) or not np.all(np.isfinite(normal_rhs)):
            if fallback is not None:
                return fallback.copy()
            raise

        scale = max(1.0, float(np.linalg.norm(normal_lhs, ord=np.inf)))
        eye = np.eye(normal_lhs.shape[0])
        for multiplier in (1.0, 1e2, 1e4, 1e6):
            try:
                return np.linalg.solve(
                    normal_lhs + ridge * multiplier * scale * eye,
                    normal_rhs,
                )
            except np.linalg.LinAlgError:
                continue

        if fallback is not None:
            return fallback.copy()
        raise


def safe_solve(
    A: np.ndarray,
    b: np.ndarray,
    fallback: np.ndarray,
    ridge: float = 1e-10,
) -> np.ndarray:
    """Solve a square linear system, increasing diagonal ridge if needed."""
    if not np.all(np.isfinite(A)) or not np.all(np.isfinite(b)):
        return fallback.copy()

    scale = max(1.0, float(np.linalg.norm(A, ord=np.inf)))
    eye = np.eye(A.shape[0])
    for multiplier in (0.0, 1.0, 1e2, 1e4, 1e6):
        try:
            lhs = A if multiplier == 0.0 else A + ridge * multiplier * scale * eye
            solution = np.linalg.solve(lhs, b)
            if np.all(np.isfinite(solution)):
                return solution
        except np.linalg.LinAlgError:
            continue

    return safe_lstsq(A, b, fallback=fallback, ridge=ridge)


def safe_inverse(
    A: np.ndarray,
    ridge: float = 1e-10,
) -> np.ndarray:
    """Invert a small square system with increasing diagonal ridge if needed."""
    if not np.all(np.isfinite(A)):
        raise np.linalg.LinAlgError("inverse input contains nonfinite values")

    scale = max(1.0, float(np.linalg.norm(A, ord=np.inf)))
    eye = np.eye(A.shape[0])
    for multiplier in (0.0, 1.0, 1e2, 1e4, 1e6):
        try:
            lhs = A if multiplier == 0.0 else A + ridge * multiplier * scale * eye
            inverse = np.linalg.inv(lhs)
            if np.all(np.isfinite(inverse)):
                return inverse
        except np.linalg.LinAlgError:
            continue
    raise np.linalg.LinAlgError("regularized inverse failed")


def lad_regression_admm(
    A: np.ndarray,
    y: np.ndarray,
    beta_init: Optional[np.ndarray] = None,
    rho: float = 1.0,
    max_iter: int = 50,
    abstol: float = 1e-5,
    reltol: float = 1e-4,
) -> np.ndarray:
    """Solve min_beta sum_i |y_i - A[i] beta| using NumPy ADMM.

    The split is r = y - A beta. The r-update is soft-thresholding, and the
    beta-update solves a fixed least-squares system for the current assigned
    samples. This is the LAD subproblem used in the robust AM refinement.
    """
    if A.ndim != 2 or y.ndim != 1:
        raise ValueError("A must be two-dimensional and y must be one-dimensional.")
    if A.shape[0] != y.shape[0]:
        raise ValueError("A and y must have the same number of rows.")

    n_samples, n_features = A.shape
    if n_samples == 0:
        if beta_init is None:
            return np.zeros(n_features)
        return beta_init.copy()

    if beta_init is None:
        beta = safe_lstsq(A, y, fallback=np.zeros(n_features))
    else:
        beta = beta_init.copy()

    r = y - A @ beta
    u = np.zeros(n_samples)
    lhs = A.T @ A
    rhs_base = A.T @ y
    lhs_scale = max(1.0, float(np.linalg.norm(lhs, ord=np.inf)))
    regularized_lhs = lhs + 1e-10 * lhs_scale * np.eye(n_features)
    try:
        beta_update_matrix = safe_inverse(regularized_lhs)
    except np.linalg.LinAlgError:
        beta_update_matrix = None

    for _ in range(max_iter):
        rhs = rhs_base - A.T @ (r + u)
        if beta_update_matrix is None:
            beta = safe_solve(regularized_lhs, rhs, fallback=beta)
        else:
            beta = beta_update_matrix @ rhs

        residual = y - A @ beta
        r_old = r
        r = soft_threshold(residual - u, 1.0 / rho)
        u = u + r - residual

        primal_norm = np.linalg.norm(r - residual)
        dual_norm = rho * np.linalg.norm(A.T @ (r - r_old))
        eps_primal = np.sqrt(n_samples) * abstol + reltol * max(
            np.linalg.norm(r), np.linalg.norm(residual)
        )
        eps_dual = np.sqrt(n_features) * abstol + reltol * np.linalg.norm(rho * A.T @ u)
        if primal_norm <= eps_primal and dual_norm <= eps_dual:
            break

    return beta


def robust_am_refine(
    X: np.ndarray,
    y: np.ndarray,
    beta0: np.ndarray,
    steps: int,
) -> np.ndarray:
    """Run several robust-AM iterations from the paper on one candidate.

    beta has shape (k, d + 1). Each step assigns observations to the largest
    affine prediction, then fits one LAD regression per assigned component.
    """
    if steps <= 0:
        return beta0.copy()

    n, d = X.shape
    k = beta0.shape[0]
    X_aug = np.column_stack((X, np.ones(n)))
    beta = beta0.copy()
    previous_assignments = None

    for _ in range(steps):
        assignments = np.argmax(X_aug @ beta.T, axis=1)
        if previous_assignments is not None and np.array_equal(assignments, previous_assignments):
            break
        previous_assignments = assignments.copy()

        beta_next = beta.copy()
        for component in range(k):
            mask = assignments == component
            if not np.any(mask):
                continue
            beta_next[component] = lad_regression_admm(
                X_aug[mask],
                y[mask],
                beta_init=beta[component],
            )
        beta = beta_next

    return beta


def _init_refinement_worker(X_aug: np.ndarray, y: np.ndarray, steps: int) -> None:
    global _REFINE_X_AUG, _REFINE_Y, _REFINE_STEPS
    _REFINE_X_AUG = X_aug
    _REFINE_Y = y
    _REFINE_STEPS = steps


def _refine_candidate_worker(beta0: np.ndarray) -> np.ndarray:
    X_aug = _REFINE_X_AUG
    y = _REFINE_Y
    if X_aug is None or y is None:
        raise RuntimeError("Refinement worker was not initialized.")

    beta = beta0.copy()
    previous_assignments = None
    for _ in range(_REFINE_STEPS):
        assignments = np.argmax(X_aug @ beta.T, axis=1)
        if previous_assignments is not None and np.array_equal(assignments, previous_assignments):
            break
        previous_assignments = assignments.copy()

        beta_next = beta.copy()
        for component in range(beta.shape[0]):
            mask = assignments == component
            if not np.any(mask):
                continue
            beta_next[component] = lad_regression_admm(
                X_aug[mask],
                y[mask],
                beta_init=beta[component],
            )
        beta = beta_next
    return beta


def refine_candidates(
    X: np.ndarray,
    y: np.ndarray,
    candidates: np.ndarray,
    steps: int,
    workers: int,
) -> np.ndarray:
    """Refine every candidate, shape (M, k, d + 1), by robust AM."""
    if steps <= 0:
        return candidates
    if workers <= 1 or candidates.shape[0] < 2 * workers:
        return np.array([robust_am_refine(X, y, candidate, steps) for candidate in candidates])

    X_aug = np.column_stack((X, np.ones(X.shape[0])))
    with mp.Pool(
        processes=workers,
        initializer=_init_refinement_worker,
        initargs=(X_aug, y, steps),
    ) as pool:
        refined = pool.map(_refine_candidate_worker, list(candidates))
    return np.asarray(refined)


def evaluate_candidates(
    X: np.ndarray,
    y: np.ndarray,
    candidates: np.ndarray,
    loss: str,
) -> Tuple[int, np.ndarray, np.ndarray]:
    """Select a candidate by fitting the nonnegative scalar c on all samples."""
    if loss not in ("l2", "l1"):
        raise ValueError("loss must be either 'l2' or 'l1'.")
    if X.ndim != 2 or y.ndim != 1 or candidates.ndim != 3:
        raise ValueError("X, y, and candidates must have shapes (n,d), (n,), (M,k,d+1).")
    n, d = X.shape
    if y.shape[0] != n or candidates.shape[2] != d + 1:
        raise ValueError("Incompatible X, y, and candidate dimensions.")

    X_aug = np.column_stack((X, np.ones(n)))
    predictions = np.max(np.einsum("nd,mkd->mnk", X_aug, candidates), axis=2)

    p_select = predictions
    y_select = y

    M = candidates.shape[0]
    c_values = np.zeros(M)
    losses = np.zeros(M)
    if loss == "l2":
        numerators = p_select @ y_select
        denominators = np.sum(p_select * p_select, axis=1)
        valid = denominators > np.finfo(float).eps
        c_values[valid] = np.maximum(0.0, numerators[valid] / denominators[valid])
        residuals = y_select[None, :] - c_values[:, None] * p_select
        losses = np.sum(residuals * residuals, axis=1)
    else:
        for index in range(M):
            c_values[index] = optimal_c_l1(y_select, p_select[index])
            residuals = y_select - c_values[index] * p_select[index]
            losses[index] = np.sum(np.abs(residuals))

    return int(np.argmin(losses)), losses, c_values


def candidate_losses(
    X: np.ndarray,
    y: np.ndarray,
    candidates: np.ndarray,
    loss: str,
) -> np.ndarray:
    """Evaluate candidate losses on all samples without refitting scale."""
    if loss not in ("l2", "l1"):
        raise ValueError("loss must be either 'l2' or 'l1'.")
    if X.ndim != 2 or y.ndim != 1 or candidates.ndim != 3:
        raise ValueError("X, y, and candidates must have shapes (n,d), (n,), (M,k,d+1).")
    n, d = X.shape
    if y.shape[0] != n or candidates.shape[2] != d + 1:
        raise ValueError("Incompatible X, y, and candidate dimensions.")

    X_aug = np.column_stack((X, np.ones(n)))
    predictions = np.max(np.einsum("nd,mkd->mnk", X_aug, candidates), axis=2)
    residuals = y[None, :] - predictions
    if loss == "l2":
        return np.sum(residuals * residuals, axis=1)
    return np.sum(np.abs(residuals), axis=1)


def low_dimensional_random_search(
    X: np.ndarray,
    y: np.ndarray,
    U_hat: np.ndarray,
    M: int,
    random_state: Optional[int] = None,
    check_orthonormal: bool = True,
    loss: str = "l2",
    refinement_steps: int = 0,
    workers: int = 1,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Run low-dimensional random search in the spectral subspace.

    Parameters
    ----------
    X : ndarray, shape (n, d)
    y : ndarray, shape (n,)
    U_hat : ndarray, shape (d, k)
        Estimated k-dimensional subspace with orthonormal columns.
    M : int
        Number of random initializations.
    random_state : int or None
    loss : {"l2", "l1"}
        Scalar fitting loss used to select the random initialization.
    refinement_steps : int
        Number of robust-AM refinement steps applied to every candidate before
        final selection. Use zero for the displayed random-search algorithm.

    Returns
    -------
    beta0 : ndarray, shape (k, d + 1)
        beta0[j] is beta_j^(0).
    info : dict
        Includes best_index, best_loss, best_c, losses, and c_values.
    """
    if loss not in ("l2", "l1"):
        raise ValueError("loss must be either 'l2' or 'l1'.")
    _, _, k = validate_random_search_inputs(X, y, U_hat, M, check_orthonormal)
    rng = np.random.default_rng(random_state)

    # nu[ell, j] is nu_j^ell, shape (M, k, k + 1).
    nu = sample_uniform_ball(rng, size=(M, k), dimension=k + 1)

    # candidates[ell, j] = V_hat nu[ell, j] = [U_hat nu[:k]; nu[k]].
    candidate_slopes = np.einsum("mjk,dk->mjd", nu[:, :, :k], U_hat)
    candidate_intercepts = nu[:, :, k:k + 1]
    candidates = np.concatenate((candidate_slopes, candidate_intercepts), axis=2)

    if refinement_steps > 0:
        candidates = refine_candidates(X, y, candidates, refinement_steps, workers)

    best_index, losses, c_values = evaluate_candidates(X, y, candidates, loss)

    beta0 = candidates[best_index]

    info = {
        "best_index": best_index,
        "best_loss": float(losses[best_index]),
        "best_c": float(c_values[best_index]),
        "losses": losses,
        "c_values": c_values,
        "loss_type": loss,
        "refinement_steps": int(refinement_steps),
    }
    return beta0, info


def full_dimensional_random_search(
    X: np.ndarray,
    y: np.ndarray,
    k: int,
    M: int,
    random_state: Optional[int] = None,
    loss: str = "l2",
    refinement_steps: int = 0,
    workers: int = 1,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Run random search directly in the full augmented parameter space."""
    if X.ndim != 2:
        raise ValueError("X must have shape (n, d).")
    if y.ndim != 1:
        raise ValueError("y must have shape (n,).")
    n, d = X.shape
    if y.shape[0] != n:
        raise ValueError("X and y must have the same number of samples.")
    if n < 2:
        raise ValueError("At least two samples are needed for the selection split.")
    if k <= 0:
        raise ValueError("k must be positive.")
    if M <= 0:
        raise ValueError("M must be positive.")

    rng = np.random.default_rng(random_state)
    candidates = sample_uniform_ball(rng, size=(M, k), dimension=d + 1)
    _, scale_losses, c_values = evaluate_candidates(X, y, candidates, loss)
    candidates = c_values[:, None, None] * candidates
    if refinement_steps > 0:
        candidates = refine_candidates(X, y, candidates, refinement_steps, workers)

    losses = candidate_losses(X, y, candidates, loss)
    best_index = int(np.argmin(losses))
    beta0 = candidates[best_index]
    info = {
        "best_index": best_index,
        "best_loss": float(losses[best_index]),
        "best_c": float(c_values[best_index]),
        "scale_loss": float(scale_losses[best_index]),
        "scale_losses": scale_losses,
        "losses": losses,
        "c_values": c_values,
        "loss_type": loss,
        "refinement_steps": int(refinement_steps),
        "mode": "nospectral",
    }
    return beta0, info


def permutation_matched_normalized_squared_error(
    beta_estimate: np.ndarray,
    beta_truth: np.ndarray,
) -> float:
    """Return min_perm ||beta_estimate[perm] - beta_truth||_F^2 / ||beta_truth||_F^2."""
    if beta_estimate.shape != beta_truth.shape:
        raise ValueError("beta_estimate and beta_truth must have the same shape.")
    truth_norm_squared = float(np.sum(beta_truth * beta_truth))
    if truth_norm_squared <= np.finfo(float).eps:
        raise ValueError("beta_truth must have nonzero Frobenius norm.")

    k = beta_truth.shape[0]
    best = np.inf
    for permutation in itertools.permutations(range(k)):
        permuted = beta_estimate[list(permutation)]
        error = float(np.sum((permuted - beta_truth) ** 2) / truth_norm_squared)
        if error < best:
            best = error
    return float(best)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run random-search initialization experiments with optional spectral subspace."
    )
    parser.add_argument(
        "--mode",
        choices=("spectral", "nospectral"),
        default="spectral",
        help="Use spectral subspace random search or full-dimensional random search.",
    )
    parser.add_argument("--nstart", type=int, default=100, help="First sample size.")
    parser.add_argument("--nend", type=int, default=1000, help="Last sample size.")
    parser.add_argument("--njump", type=int, default=100, help="Sample size step.")
    parser.add_argument("--m", type=int, default=10, help="Monte Carlo runs per sample size.")
    parser.add_argument("--d", type=int, default=5, help="Covariate dimension.")
    parser.add_argument("--k", type=int, default=2, help="Number of max-affine pieces.")
    parser.add_argument("--alpha", type=float, default=0.1, help="Outlier fraction.")
    parser.add_argument("--M", type=float, default=5.0, help="Uniform outlier bound.")
    parser.add_argument(
        "--Mrand",
        type=int,
        default=200,
        help="Number of random-search initializations.",
    )
    parser.add_argument(
        "--refinement-steps",
        type=int,
        default=0,
        help="Robust-AM refinement steps applied to every sampled candidate.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(28, max(1, os.cpu_count() or 1)),
        help="Parallel worker processes for candidate refinement.",
    )
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
    if args.Mrand <= 0:
        raise ValueError("Mrand must be positive.")
    if args.refinement_steps < 0:
        raise ValueError("refinement-steps must be nonnegative.")
    if args.workers <= 0:
        raise ValueError("workers must be positive.")
    if args.d <= 0:
        raise ValueError("d must be positive.")
    if args.k <= 0:
        raise ValueError("k must be positive.")
    if args.k > args.d:
        raise ValueError("k must be no larger than d.")
    if not 0 <= args.alpha < 1:
        raise ValueError("alpha must be at least 0 and less than 1.")
    if args.M <= 0:
        raise ValueError("M must be positive.")


def sample_sizes(args: argparse.Namespace) -> np.ndarray:
    return np.arange(args.nstart, args.nend + 1, args.njump, dtype=int)


def result_filename(args: argparse.Namespace) -> str:
    seed_value = "none" if args.seed is None else args.seed
    parts = [
        "random_search",
        args.mode,
        f"k{args.k}",
        f"d{args.d}",
        f"alpha{format_value(args.alpha)}",
        f"M{format_value(args.M)}",
        f"Mrand{args.Mrand}",
        f"ref{args.refinement_steps}",
        f"n{args.nstart}-{args.nend}-{args.njump}",
        f"m{args.m}",
        f"seed{seed_value}",
    ]
    return "_".join(parts) + ".npz"


def run_trial(
    n: int,
    args: argparse.Namespace,
    rng: Any,
) -> Tuple[float, float, float, float, float, float, float, float]:
    beta = sample_max_affine_vectors(args.k, args.d, rng=rng)
    x, clean_y = generate_clean_dataset(beta, n, rng)
    contaminated_y, outlier_indices = contaminate_responses(clean_y, args.alpha, args.M, rng)
    clean_mask = np.ones(n, dtype=bool)
    clean_mask[outlier_indices] = False

    if args.mode == "spectral":
        U_total = spectral_subspace(x, contaminated_y, args.k)
        U_clean = spectral_subspace(x[clean_mask], clean_y[clean_mask], args.k)

        beta_total_l2, info_total_l2 = low_dimensional_random_search(
            x,
            contaminated_y,
            U_total,
            args.Mrand,
            random_state=rng,
            loss="l2",
            refinement_steps=args.refinement_steps,
            workers=args.workers,
        )
        beta_total_l1, info_total_l1 = low_dimensional_random_search(
            x,
            contaminated_y,
            U_total,
            args.Mrand,
            random_state=rng,
            loss="l1",
            refinement_steps=args.refinement_steps,
            workers=args.workers,
        )
        beta_clean_l2, info_clean_l2 = low_dimensional_random_search(
            x[clean_mask],
            clean_y[clean_mask],
            U_clean,
            args.Mrand,
            random_state=rng,
            loss="l2",
            refinement_steps=args.refinement_steps,
            workers=args.workers,
        )
        beta_clean_l1, info_clean_l1 = low_dimensional_random_search(
            x[clean_mask],
            clean_y[clean_mask],
            U_clean,
            args.Mrand,
            random_state=rng,
            loss="l1",
            refinement_steps=args.refinement_steps,
            workers=args.workers,
        )
    else:
        beta_total_l2, info_total_l2 = full_dimensional_random_search(
            x,
            contaminated_y,
            args.k,
            args.Mrand,
            random_state=rng,
            loss="l2",
            refinement_steps=args.refinement_steps,
            workers=args.workers,
        )
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
        beta_clean_l2, info_clean_l2 = full_dimensional_random_search(
            x[clean_mask],
            clean_y[clean_mask],
            args.k,
            args.Mrand,
            random_state=rng,
            loss="l2",
            refinement_steps=args.refinement_steps,
            workers=args.workers,
        )
        beta_clean_l1, info_clean_l1 = full_dimensional_random_search(
            x[clean_mask],
            clean_y[clean_mask],
            args.k,
            args.Mrand,
            random_state=rng,
            loss="l1",
            refinement_steps=args.refinement_steps,
            workers=args.workers,
        )

    def reported_beta(beta_hat: np.ndarray, info: Dict[str, Any]) -> np.ndarray:
        if info.get("mode") == "nospectral":
            return beta_hat
        return float(info["best_c"]) * beta_hat

    total_l2_error = permutation_matched_normalized_squared_error(
        reported_beta(beta_total_l2, info_total_l2),
        beta,
    )
    total_l1_error = permutation_matched_normalized_squared_error(
        reported_beta(beta_total_l1, info_total_l1),
        beta,
    )
    clean_l2_error = permutation_matched_normalized_squared_error(
        reported_beta(beta_clean_l2, info_clean_l2),
        beta,
    )
    clean_l1_error = permutation_matched_normalized_squared_error(
        reported_beta(beta_clean_l1, info_clean_l1),
        beta,
    )
    return (
        total_l2_error,
        total_l1_error,
        clean_l2_error,
        clean_l1_error,
        info_total_l2["best_c"],
        info_total_l1["best_c"],
        info_clean_l2["best_c"],
        info_clean_l1["best_c"],
    )


def save_results(
    args: argparse.Namespace,
    n_values: np.ndarray,
    total_l2_errors: np.ndarray,
    total_l1_errors: np.ndarray,
    clean_l2_errors: np.ndarray,
    clean_l1_errors: np.ndarray,
    total_l2_c_values: np.ndarray,
    total_l1_c_values: np.ndarray,
    clean_l2_c_values: np.ndarray,
    clean_l1_c_values: np.ndarray,
) -> Path:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / result_filename(args)
    metadata = {
        "nstart": args.nstart,
        "mode": args.mode,
        "nend": args.nend,
        "njump": args.njump,
        "m": args.m,
        "d": args.d,
        "k": args.k,
        "alpha": args.alpha,
        "M": args.M,
        "Mrand": args.Mrand,
        "refinement_steps": args.refinement_steps,
        "workers": args.workers,
        "seed": args.seed,
        "metric": "permutation_matched_normalized_squared_error",
    }
    np.savez(
        str(output_path),
        n_values=n_values,
        total_l2_errors=total_l2_errors,
        total_l1_errors=total_l1_errors,
        clean_l2_errors=clean_l2_errors,
        clean_l1_errors=clean_l1_errors,
        total_l2_c_values=total_l2_c_values,
        total_l1_c_values=total_l1_c_values,
        clean_l2_c_values=clean_l2_c_values,
        clean_l1_c_values=clean_l1_c_values,
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
    total_l2_errors = np.zeros((len(n_values), args.m))
    total_l1_errors = np.zeros((len(n_values), args.m))
    clean_l2_errors = np.zeros((len(n_values), args.m))
    clean_l1_errors = np.zeros((len(n_values), args.m))
    total_l2_c_values = np.zeros((len(n_values), args.m))
    total_l1_c_values = np.zeros((len(n_values), args.m))
    clean_l2_c_values = np.zeros((len(n_values), args.m))
    clean_l1_c_values = np.zeros((len(n_values), args.m))

    for n_index, n in enumerate(n_values):
        for run_index in range(args.m):
            print_progress(n_index, len(n_values), int(n), run_index, args.m)
            (
                total_l2_error,
                total_l1_error,
                clean_l2_error,
                clean_l1_error,
                total_l2_c,
                total_l1_c,
                clean_l2_c,
                clean_l1_c,
            ) = run_trial(int(n), args, rng)
            total_l2_errors[n_index, run_index] = total_l2_error
            total_l1_errors[n_index, run_index] = total_l1_error
            clean_l2_errors[n_index, run_index] = clean_l2_error
            clean_l1_errors[n_index, run_index] = clean_l1_error
            total_l2_c_values[n_index, run_index] = total_l2_c
            total_l1_c_values[n_index, run_index] = total_l1_c
            clean_l2_c_values[n_index, run_index] = clean_l2_c
            clean_l1_c_values[n_index, run_index] = clean_l1_c

    output_path = save_results(
        args,
        n_values,
        total_l2_errors,
        total_l1_errors,
        clean_l2_errors,
        clean_l1_errors,
        total_l2_c_values,
        total_l1_c_values,
        clean_l2_c_values,
        clean_l1_c_values,
    )
    print("\r\033[K", end="")
    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        rng_test = np.random.default_rng(0)
        X_test = rng_test.standard_normal(size=(20, 4))
        U_test, _ = np.linalg.qr(rng_test.standard_normal(size=(4, 2)))
        y_test = rng_test.standard_normal(size=20)
        beta0_test, info_test = low_dimensional_random_search(
            X_test,
            y_test,
            U_test,
            M=5,
            random_state=0,
        )
        c_l1_test = optimal_c_l1(np.array([1.0, 2.0, 3.0]), np.array([1.0, 1.0, 2.0]))
        beta0_l1_test, info_l1_test = low_dimensional_random_search(
            X_test,
            y_test,
            U_test,
            M=5,
            random_state=1,
            loss="l1",
        )
        print("Synthetic test optimal_c_l1:", c_l1_test)
        print("Synthetic test beta0 shape:", beta0_test.shape)
        print("Synthetic test best loss:", info_test["best_loss"])
        print("Synthetic test L1 beta0 shape:", beta0_l1_test.shape)
        print("Synthetic test L1 best loss:", info_l1_test["best_loss"])
    else:
        main()
