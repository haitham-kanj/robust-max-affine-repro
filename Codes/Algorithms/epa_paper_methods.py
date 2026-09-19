"""Replicate the three EPA real-data methods reported by Blanchet et al. (2020).

Methods:
    DRCR: finite-dimensional LP from equation (7), using absolute loss and
          delta * max_i ||xi_i||_inf regularization.
    LSE:  least-squares convex regression over the same finite-dimensional
          convexity constraints, with ||xi_i||_inf <= c.
    LR:   ordinary least-squares linear regression.

The convex programs require cvxpy. The data loading and preprocessing mirror
epa_real_data_k_sweep.py: positive rows only, log covariates, then standardize.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Codes.Algorithms.epa_real_data_k_sweep import (  # noqa: E402
    DATA_DIR,
    RESULTS_DIR,
    build_design,
    download_rows,
    read_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run DRCR, LSE, and LR on the EPA dataset using the paper's protocol."
    )
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--api-key", default=os.environ.get("EPA_API_KEY"))
    parser.add_argument("--year", type=int, default=2019)
    parser.add_argument("--quarter", type=int, default=1)
    parser.add_argument("--state-code", default="CA")
    parser.add_argument(
        "--aggregation",
        choices=("daily", "hourly", "monthly", "quarterly"),
        default="daily",
    )
    parser.add_argument("--per-page", type=int, default=500)
    parser.add_argument("--max-pages", type=int, default=200)
    parser.add_argument("--max-usable-rows", type=int, default=600)
    parser.add_argument("--train-size", type=int, default=400)
    parser.add_argument("--test-size", type=int, default=200)
    parser.add_argument("--splits", type=int, default=10)
    parser.add_argument(
        "--delta",
        type=float,
        default=None,
        help="DRCR regularization. Defaults to n_train^(-2/d), as in the paper.",
    )
    parser.add_argument(
        "--c",
        type=float,
        default=10.0,
        help="LSE gradient infinity-norm bound ||xi_i||_inf <= c.",
    )
    parser.add_argument(
        "--solver",
        default=None,
        help="Optional cvxpy solver name, e.g. CLARABEL, ECOS, OSQP, SCS.",
    )
    parser.add_argument(
        "--methods",
        default="DRCR,LSE,LR",
        help="Comma-separated subset of DRCR,LSE,LR.",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    return parser.parse_args()


def import_cvxpy():
    try:
        import cvxpy as cp
    except ImportError as error:
        raise RuntimeError(
            "DRCR and LSE require cvxpy. Install it in the active environment, "
            "then rerun this script."
        ) from error
    return cp


def affine_predictions(
    X_eval: np.ndarray,
    X_train: np.ndarray,
    g: np.ndarray,
    xi: np.ndarray,
) -> np.ndarray:
    slopes = np.sum((X_eval[:, None, :] - X_train[None, :, :]) * xi[None, :, :], axis=2)
    return np.max(g[None, :] + slopes, axis=1)


def convexity_constraints(cp: Any, X: np.ndarray, g: Any, xi: Any) -> Iterable[Any]:
    n = X.shape[0]
    constraints = []
    for i in range(n):
        differences = X - X[i]
        constraints.append(g >= g[i] + differences @ xi[i, :])
    return constraints


def solve_drcr(
    X: np.ndarray,
    y: np.ndarray,
    delta: float,
    solver: Optional[str],
) -> Tuple[np.ndarray, np.ndarray, float]:
    cp = import_cvxpy()
    n, d = X.shape
    g = cp.Variable(n)
    xi = cp.Variable((n, d))
    residual = cp.Variable(n, nonneg=True)
    lip_bound = np.log(n)
    t = cp.Variable(nonneg=True)

    constraints = list(convexity_constraints(cp, X, g, xi))
    constraints += [
        residual >= y - g,
        residual >= g - y,
        xi <= lip_bound,
        xi >= -lip_bound,
        xi <= t,
        xi >= -t,
        t <= lip_bound,
    ]
    objective = cp.Minimize(cp.sum(residual) / n + delta * t)
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=solver, verbose=False)
    if problem.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"DRCR solve failed with status {problem.status}.")
    return np.asarray(g.value).reshape(n), np.asarray(xi.value).reshape(n, d), float(problem.value)


def solve_lse(
    X: np.ndarray,
    y: np.ndarray,
    c: float,
    solver: Optional[str],
) -> Tuple[np.ndarray, np.ndarray, float]:
    cp = import_cvxpy()
    n, d = X.shape
    g = cp.Variable(n)
    xi = cp.Variable((n, d))

    constraints = list(convexity_constraints(cp, X, g, xi))
    constraints += [xi <= c, xi >= -c]
    objective = cp.Minimize(cp.sum_squares(y - g) / n)
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=solver, verbose=False)
    if problem.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"LSE solve failed with status {problem.status}.")
    return np.asarray(g.value).reshape(n), np.asarray(xi.value).reshape(n, d), float(problem.value)


def fit_lr(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    X_aug = np.column_stack((X, np.ones(X.shape[0])))
    beta, *_ = np.linalg.lstsq(X_aug, y, rcond=None)
    return beta


def predict_lr(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    X_aug = np.column_stack((X, np.ones(X.shape[0])))
    return X_aug @ beta


def mean_l1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def run_split(
    X: np.ndarray,
    y: np.ndarray,
    args: argparse.Namespace,
    rng: np.random.Generator,
    methods: Tuple[str, ...],
) -> Dict[str, Tuple[float, float]]:
    permutation = rng.permutation(X.shape[0])
    train_indices = permutation[: args.train_size]
    test_indices = permutation[args.train_size: args.train_size + args.test_size]
    X_train = X[train_indices]
    y_train = y[train_indices]
    X_test = X[test_indices]
    y_test = y[test_indices]
    delta = args.delta if args.delta is not None else args.train_size ** (-2.0 / X.shape[1])

    results: Dict[str, Tuple[float, float]] = {}

    if "DRCR" in methods:
        g_drcr, xi_drcr, _ = solve_drcr(X_train, y_train, delta, args.solver)
        results["DRCR"] = (
            mean_l1(y_train, affine_predictions(X_train, X_train, g_drcr, xi_drcr)),
            mean_l1(y_test, affine_predictions(X_test, X_train, g_drcr, xi_drcr)),
        )

    if "LSE" in methods:
        g_lse, xi_lse, _ = solve_lse(X_train, y_train, args.c, args.solver)
        results["LSE"] = (
            mean_l1(y_train, affine_predictions(X_train, X_train, g_lse, xi_lse)),
            mean_l1(y_test, affine_predictions(X_test, X_train, g_lse, xi_lse)),
        )

    if "LR" in methods:
        beta_lr = fit_lr(X_train, y_train)
        results["LR"] = (
            mean_l1(y_train, predict_lr(X_train, beta_lr)),
            mean_l1(y_test, predict_lr(X_test, beta_lr)),
        )
    return results


def load_data(args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray, Path]:
    if args.download:
        rows, data_path = download_rows(args)
    elif args.data_path is not None:
        data_path = args.data_path
        rows = read_rows(data_path)
    else:
        cached = sorted(DATA_DIR.glob(f"campd_*_{args.state_code.lower()}_q{args.quarter}_{args.year}.*"))
        if not cached:
            raise RuntimeError("No cached data found. Run with --download or pass --data-path.")
        data_path = cached[0]
        rows = read_rows(data_path)

    rng = np.random.default_rng(args.seed)
    X, y = build_design(rows, args.max_usable_rows, rng)
    if args.train_size + args.test_size > X.shape[0]:
        raise ValueError(
            f"Need {args.train_size + args.test_size} rows but only found {X.shape[0]} usable rows."
        )
    return X, y, data_path


def main() -> None:
    args = parse_args()
    X, y, data_path = load_data(args)
    rng = np.random.default_rng(args.seed)
    methods = tuple(method.strip().upper() for method in args.methods.split(",") if method.strip())
    valid_methods = {"DRCR", "LSE", "LR"}
    if not methods or any(method not in valid_methods for method in methods):
        raise ValueError("--methods must be a comma-separated subset of DRCR,LSE,LR.")
    train_errors = {method: [] for method in methods}
    test_errors = {method: [] for method in methods}

    print(f"Using {X.shape[0]} usable rows from {data_path} with d={X.shape[1]}.")
    print(f"DRCR delta={args.delta if args.delta is not None else args.train_size ** (-2.0 / X.shape[1]):.6g}; LSE c={args.c:g}")

    for split in range(args.splits):
        split_results = run_split(X, y, args, rng, methods)
        pieces = []
        for method in methods:
            train_error, test_error = split_results[method]
            train_errors[method].append(train_error)
            test_errors[method].append(test_error)
            pieces.append(f"{method}: train={train_error:.4f}, test={test_error:.4f}")
        print(f"split={split + 1}/{args.splits}: " + "; ".join(pieces), flush=True)

    print("Averages:")
    for method in methods:
        print(
            f"{method}: mean_train_l1={np.mean(train_errors[method]):.4f}, "
            f"mean_test_l1={np.mean(test_errors[method]):.4f}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / (
        f"epa_paper_methods_{args.aggregation}_splits{args.splits}_seed{args.seed}.npz"
    )
    metadata = {
        "dataset": "EPA CAMPD California Q1 2019",
        "aggregation": args.aggregation,
        "data_path": str(data_path),
        "preprocessing": "positive rows only; log covariates; standardize X and y",
        "train_size": args.train_size,
        "test_size": args.test_size,
        "splits": args.splits,
        "seed": args.seed,
        "drcr_delta": args.delta if args.delta is not None else args.train_size ** (-2.0 / X.shape[1]),
        "lse_c": args.c,
        "solver": args.solver,
        "methods": methods,
    }
    np.savez(
        str(output_path),
        methods=np.asarray(methods),
        train_errors=np.asarray([train_errors[method] for method in methods]),
        test_errors=np.asarray([test_errors[method] for method in methods]),
        metadata=json.dumps(metadata, indent=2, sort_keys=True),
    )
    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
