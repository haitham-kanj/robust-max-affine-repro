"""Multi-GPU version of the total-L1 nospectral sweeper.

This script produces the same result artifact as Codes/Algorithms/sweeper.py:
results/sweeper/sweeper_total_l1_nospectral_*.npz with the same array keys.
It parallelizes independent Monte Carlo cells across GPUs and keeps the
candidate refinement computations on-device with CuPy.
"""

import argparse
import json
import os
import queue
import sys
import time
from multiprocessing import Process, Queue
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Codes.Algorithms.random_search import (  # noqa: E402
    permutation_matched_normalized_squared_error,
    sample_uniform_ball,
)
from Codes.Algorithms.spectral import (  # noqa: E402
    contaminate_responses,
    format_value,
    generate_clean_dataset,
)
from Codes.Algorithms.sweeper import (  # noqa: E402
    grid_values,
    load_existing_results,
    result_filename,
    save_results,
    validate_args,
)
from Codes.utilities import sample_max_affine_vectors  # noqa: E402


RESULTS_DIR = PROJECT_ROOT / "results" / "sweeper"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the total-L1 nospectral sweeper across multiple GPUs."
    )
    parser.add_argument("--nstart", type=int, default=100)
    parser.add_argument("--nend", type=int, default=2500)
    parser.add_argument("--njump", type=int, default=200)
    parser.add_argument("--dstart", type=int, default=20)
    parser.add_argument("--dend", type=int, default=100)
    parser.add_argument("--djump", type=int, default=10)
    parser.add_argument("--m", type=int, default=20)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--M", type=float, default=50.0)
    parser.add_argument("--Mrand", type=int, default=20)
    parser.add_argument("--refinement-steps", type=int, default=5)
    parser.add_argument("--max-iterations", type=int, default=50)
    parser.add_argument("--threshold", type=float, default=1e-3)
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Kept for metadata compatibility; GPU work uses --gpus.",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--gpus",
        default="0,1,2,3",
        help="Comma-separated GPU ids visible inside the Slurm allocation.",
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "float64"),
        default="float32",
        help="float32 is faster on A100 and usually enough for this Monte Carlo search.",
    )
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    return parser.parse_args()


def parse_gpus(value: str) -> List[str]:
    gpus = [item.strip() for item in value.split(",") if item.strip()]
    if not gpus:
        raise ValueError("--gpus must contain at least one GPU id.")
    return gpus


def soft_threshold_cp(cp: Any, values: Any, threshold: float) -> Any:
    return cp.sign(values) * cp.maximum(cp.abs(values) - threshold, 0.0)


def batched_lad_regression_admm_cp(
    cp: Any,
    A: Any,
    y: Any,
    assignments: Any,
    beta_init: Any,
    candidate_active: Any,
    max_iter: int = 50,
    abstol: float = 1e-5,
    reltol: float = 1e-4,
) -> Any:
    """Fit every candidate/component LAD model in one batched GPU operation."""
    candidate_count, component_count, feature_count = beta_init.shape
    component_ids = cp.arange(component_count)[None, :, None]
    mask = assignments[:, None, :] == component_ids
    counts = cp.sum(mask, axis=2)
    valid = (counts > 0) & candidate_active[:, None]
    mask_float = mask.astype(A.dtype)

    lhs = cp.einsum("mkn,np,nq->mkpq", mask_float, A, A, optimize=True)
    rhs_base = cp.einsum("mkn,n,np->mkp", mask_float, y, A, optimize=True)
    lhs_scale = cp.maximum(1.0, cp.max(cp.sum(cp.abs(lhs), axis=-1), axis=-1))
    eye = cp.eye(feature_count, dtype=A.dtype)[None, None, :, :]
    ridge = 1e-6 if A.dtype == cp.float32 else 1e-10
    regularized_lhs = lhs + (ridge * lhs_scale[..., None, None]) * eye
    inverse = cp.linalg.inv(regularized_lhs)

    beta = beta_init.copy()
    predictions = cp.einsum("np,mkp->mkn", A, beta, optimize=True)
    y_batched = y[None, None, :]
    r = mask_float * (y_batched - predictions)
    u = cp.zeros_like(r)
    group_active = valid.copy()
    sqrt_counts = cp.sqrt(counts.astype(A.dtype))
    sqrt_features = np.sqrt(feature_count)

    for _ in range(max_iter):
        rhs = rhs_base - cp.einsum(
            "mkn,np->mkp", mask_float * (r + u), A, optimize=True
        )
        beta_update = cp.einsum("mkpq,mkq->mkp", inverse, rhs, optimize=True)
        beta = cp.where(group_active[..., None], beta_update, beta)

        residual = mask_float * (
            y_batched - cp.einsum("np,mkp->mkn", A, beta, optimize=True)
        )
        r_old = r
        r_update = mask_float * soft_threshold_cp(cp, residual - u, 1.0)
        u_update = mask_float * (u + r_update - residual)
        active_rows = group_active[..., None]
        r = cp.where(active_rows, r_update, r)
        u = cp.where(active_rows, u_update, u)

        primal_norm = cp.linalg.norm(r - residual, axis=2)
        dual_vector = cp.einsum(
            "mkn,np->mkp", mask_float * (r - r_old), A, optimize=True
        )
        dual_norm = cp.linalg.norm(dual_vector, axis=2)
        eps_primal = sqrt_counts * abstol + reltol * cp.maximum(
            cp.linalg.norm(r, axis=2), cp.linalg.norm(residual, axis=2)
        )
        atu = cp.einsum("mkn,np->mkp", mask_float * u, A, optimize=True)
        eps_dual = sqrt_features * abstol + reltol * cp.linalg.norm(atu, axis=2)
        converged = (primal_norm <= eps_primal) & (dual_norm <= eps_dual)
        group_active &= ~converged

    return cp.where(valid[..., None], beta, beta_init)


def optimal_c_l1_np(y: np.ndarray, p: np.ndarray, eps: float = 1e-12) -> float:
    mask = np.abs(p) > eps
    if not np.any(mask):
        return 0.0
    ratios = y[mask] / p[mask]
    weights = np.abs(p[mask])
    order = np.argsort(ratios)
    cumulative_weights = np.cumsum(weights[order])
    half_weight = 0.5 * weights.sum()
    index = int(np.searchsorted(cumulative_weights, half_weight, side="left"))
    return float(max(0.0, ratios[order[min(index, ratios.shape[0] - 1)]]))


def full_dimensional_random_search_gpu(
    cp: Any,
    X_np: np.ndarray,
    y_np: np.ndarray,
    k: int,
    M: int,
    rng: Any,
    refinement_steps: int,
    dtype: Any,
    progress: Optional[Callable[[int], None]] = None,
) -> Tuple[np.ndarray, float]:
    n, d = X_np.shape
    candidates_np = sample_uniform_ball(rng, size=(M, k), dimension=d + 1).astype(np.float64)
    X_aug = cp.asarray(np.column_stack((X_np, np.ones(n))), dtype=dtype)
    y = cp.asarray(y_np, dtype=dtype)
    candidates = cp.asarray(candidates_np, dtype=dtype)

    raw_predictions = cp.max(cp.einsum("nd,mkd->mnk", X_aug, candidates), axis=2)
    raw_predictions_np = cp.asnumpy(raw_predictions)
    y_select = y_np
    c_values = np.zeros(M)
    for index in range(M):
        c_values[index] = optimal_c_l1_np(y_select, raw_predictions_np[index])
    candidates *= cp.asarray(c_values, dtype=dtype)[:, None, None]

    previous_assignments = None
    candidate_active = cp.ones(M, dtype=cp.bool_)
    for refinement_index in range(refinement_steps):
        predictions = cp.einsum("nd,mkd->mnk", X_aug, candidates)
        assignments = cp.argmax(predictions, axis=2)
        if previous_assignments is not None:
            candidate_active &= cp.any(assignments != previous_assignments, axis=1)
        previous_assignments = assignments
        candidates = batched_lad_regression_admm_cp(
            cp,
            X_aug,
            y,
            assignments,
            candidates,
            candidate_active,
        )
        if progress is not None:
            progress(refinement_index + 1)

    predictions = cp.max(cp.einsum("nd,mkd->mnk", X_aug, candidates), axis=2)
    predictions_np = cp.asnumpy(predictions)
    losses = np.sum(np.abs(y_select[None, :] - predictions_np), axis=1)
    best_index = int(np.argmin(losses))
    best_candidate = cp.asnumpy(candidates[best_index]).astype(np.float64)
    if not np.all(np.isfinite(best_candidate)) or not np.isfinite(losses[best_index]):
        raise FloatingPointError("Batched GPU refinement produced a nonfinite candidate.")
    return best_candidate, float(c_values[best_index])


def run_attempt_gpu(
    cp: Any,
    n: int,
    d: int,
    args: argparse.Namespace,
    rng: Any,
    dtype: Any,
    progress: Optional[Callable[[int], None]] = None,
) -> Tuple[float, float, int, bool, np.ndarray]:
    beta = sample_max_affine_vectors(args.k, d, rng=rng)
    x, clean_y = generate_clean_dataset(beta, n, rng)
    contaminated_y, _ = contaminate_responses(clean_y, args.alpha, args.M, rng)
    beta_total_l1, c_value = full_dimensional_random_search_gpu(
        cp,
        x,
        contaminated_y,
        args.k,
        args.Mrand,
        rng,
        args.refinement_steps,
        dtype,
        progress,
    )
    selected = cp.asarray(beta_total_l1, dtype=dtype)[None, :, :]
    x_aug = cp.asarray(np.column_stack((x, np.ones(n))), dtype=dtype)
    y_gpu = cp.asarray(contaminated_y, dtype=dtype)
    previous_assignments = None
    iterations = 0

    for iteration in range(args.max_iterations):
        assignments = cp.argmax(cp.einsum("nd,mkd->mnk", x_aug, selected), axis=2)
        if previous_assignments is not None and bool(
            cp.all(assignments == previous_assignments).item()
        ):
            break
        previous_assignments = assignments.copy()
        selected = batched_lad_regression_admm_cp(
            cp,
            x_aug,
            y_gpu,
            assignments,
            selected,
            cp.ones(1, dtype=cp.bool_),
        )
        iterations = iteration + 1
        if progress is not None:
            progress(args.refinement_steps + iterations)

    final_beta = cp.asnumpy(selected[0]).astype(np.float64)
    error = permutation_matched_normalized_squared_error(final_beta, beta)
    hit_threshold = error < args.threshold
    return float(error), float(c_value), iterations, hit_threshold, final_beta


def run_cell_gpu(
    cp: Any,
    n: int,
    d: int,
    args: argparse.Namespace,
    seed: int,
    dtype: Any,
    progress: Optional[Callable[[int, int], None]] = None,
) -> Tuple[float, float, int, bool, np.ndarray]:
    rng = np.random.default_rng(seed)
    return run_attempt_gpu(
        cp,
        n,
        d,
        args,
        rng,
        dtype,
        lambda refinement: progress(1, refinement) if progress else None,
    )


def worker_main(
    gpu_id: str,
    task_queue: Queue,
    result_queue: Queue,
    args: argparse.Namespace,
) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    try:
        import cupy as cp
    except ImportError as error:
        result_queue.put(("error", gpu_id, "CuPy is not installed: %s" % error))
        return

    dtype = cp.float32 if args.dtype == "float32" else cp.float64
    try:
        with cp.cuda.Device(0):
            while True:
                task = task_queue.get()
                if task is None:
                    break
                d_index, n_index, run_index, n, d, seed = task
                start = time.perf_counter()
                try:
                    result = run_cell_gpu(
                        cp,
                        n,
                        d,
                        args,
                        seed,
                        dtype,
                    )
                    cp.cuda.Stream.null.synchronize()
                    used_bytes = int(cp.get_default_memory_pool().used_bytes())
                    result_queue.put((
                        "result",
                        gpu_id,
                        d_index,
                        n_index,
                        run_index,
                        result,
                        time.perf_counter() - start,
                        used_bytes,
                    ))
                except Exception as error:
                    result_queue.put(("error", gpu_id, repr(error)))
                    break
    finally:
        try:
            cp.get_default_memory_pool().free_all_blocks()
        except Exception:
            pass


def task_seed(root_seed: int, d_index: int, n_index: int, run_index: int) -> int:
    seed_seq = np.random.SeedSequence([root_seed, d_index, n_index, run_index])
    return int(seed_seq.generate_state(1, dtype=np.uint32)[0])


def main() -> None:
    args = parse_args()
    validate_args(args)
    gpus = parse_gpus(args.gpus)
    args.workers = len(gpus)

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

    tasks = []
    for d_index, d in enumerate(d_values):
        for n_index, n in enumerate(n_values):
            for run_index in range(args.m):
                if iterations_used[d_index, n_index, run_index] > 0:
                    continue
                tasks.append((
                    d_index,
                    n_index,
                    run_index,
                    int(n),
                    int(d),
                    task_seed(args.seed, d_index, n_index, run_index),
                ))

    if not tasks:
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
        return

    task_queue = Queue()
    result_queue = Queue()
    for task in tasks:
        task_queue.put(task)
    for _ in gpus:
        task_queue.put(None)

    workers = [
        Process(target=worker_main, args=(gpu_id, task_queue, result_queue, args))
        for gpu_id in gpus
    ]
    for process in workers:
        process.start()

    total = int(np.prod(shape))
    completed = total - len(tasks)
    max_gpu_memory = {gpu_id: 0 for gpu_id in gpus}
    try:
        while completed < total:
            try:
                message = result_queue.get(timeout=30)
            except queue.Empty:
                alive = any(process.is_alive() for process in workers)
                if not alive:
                    raise RuntimeError("All GPU workers exited before completing the sweep.")
                continue

            if message[0] == "error":
                raise RuntimeError("GPU %s worker failed: %s" % (message[1], message[2]))

            _, gpu_id, d_index, n_index, run_index, result, elapsed, used_bytes = message
            error, c_value, attempts, hit_threshold, beta = result
            best_errors[d_index, n_index, run_index] = error
            best_c_values[d_index, n_index, run_index] = c_value
            iterations_used[d_index, n_index, run_index] = attempts
            stopped_early[d_index, n_index, run_index] = hit_threshold
            beta_width = beta.shape[1]
            best_betas[d_index, n_index, run_index, :, :beta_width] = beta
            best_beta_masks[d_index, n_index, run_index, :, :beta_width] = True
            max_gpu_memory[gpu_id] = max(max_gpu_memory[gpu_id], int(used_bytes))
            completed += 1

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
            message = "n=%d, d=%d, monte=%d/%d, Global progress=%.2f%%" % (
                int(n_values[n_index]),
                int(d_values[d_index]),
                run_index + 1,
                args.m,
                100.0 * completed / total,
            )
            print("\r\033[K" + message, end="", flush=True)
    finally:
        print()
        for process in workers:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()

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
    metadata_path = output_path.with_suffix(".gpu.json")
    metadata_path.write_text(
        json.dumps(
            {
                "gpus": gpus,
                "dtype": args.dtype,
                "max_gpu_memory_bytes": max_gpu_memory,
                "output": str(output_path),
                "note": "NPZ keys and filename match Codes/Algorithms/sweeper.py.",
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
