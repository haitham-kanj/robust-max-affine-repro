"""EPA real-data k sweep for total-L1 nospectral max-affine regression.

The data description follows Blanchet et al. (2020): California EPA air-market
data from Q1 2019, with heat input as the response and SO2, NOx, CO2, and NOx
rate as covariates. The script fetches CAMPD data when an EPA API key is
available, or reads a cached CSV/JSON export. By default it downloads daily Q1
rows and keeps a reproducible 600-row subsample, matching the paper's split
sizes. Covariates are log-transformed, then X and y are standardized before
repeated 400/200 train/test splits.
"""

import argparse
import csv
import io
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Codes.Algorithms.random_search import full_dimensional_random_search  # noqa: E402


RESULTS_DIR = PROJECT_ROOT / "results" / "real_data"
DATA_DIR = PROJECT_ROOT / "data" / "epa"
EMISSIONS_BASE_URL = "https://api.epa.gov/easey/emissions-mgmt/emissions/apportioned"
LEGACY_QUARTERLY_ENDPOINTS = (
    "https://api.epa.gov/easey/streaming-services/emissions/apportioned/quarterly",
    "https://api.epa.gov/easey/streaming-services/emissions/apportioned/quarterly/by-facility",
)

HEAT_INPUT_KEYS = ("heatInput", "heatInputMeasure", "heatInputSum", "heatInputQuantity")
SO2_KEYS = ("so2Mass", "so2MassMeasure", "so2MassSum", "so2MassQuantity")
NOX_KEYS = ("noxMass", "noxMassMeasure", "noxMassSum", "noxMassQuantity")
CO2_KEYS = ("co2Mass", "co2MassMeasure", "co2MassSum", "co2MassQuantity")
NOX_RATE_KEYS = ("noxRate", "noxRateMeasure", "noxRateAverage", "noxRateQuantity")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep k on EPA California Q1 2019 data using total-L1 nospectral search."
    )
    parser.add_argument("--data-path", type=Path, default=None, help="Cached CAMPD CSV/JSON file.")
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download CAMPD data into data/epa before running.",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("EPA_API_KEY"),
        help="EPA API key. Defaults to EPA_API_KEY.",
    )
    parser.add_argument("--year", type=int, default=2019)
    parser.add_argument("--quarter", type=int, default=1)
    parser.add_argument("--state-code", default="CA")
    parser.add_argument(
        "--aggregation",
        choices=("daily", "hourly", "monthly", "quarterly"),
        default="daily",
        help="CAMPD aggregation to download. Daily is the default because quarterly/monthly do not leave 600 usable positive rows.",
    )
    parser.add_argument("--per-page", type=int, default=500)
    parser.add_argument(
        "--max-pages",
        type=int,
        default=200,
        help="Maximum pages to fetch from paginated CAMPD endpoints.",
    )
    parser.add_argument(
        "--max-usable-rows",
        type=int,
        default=600,
        help="Reproducibly subsample this many usable rows after preprocessing; use 0 to keep all usable rows.",
    )
    parser.add_argument("--train-size", type=int, default=400)
    parser.add_argument("--test-size", type=int, default=200)
    parser.add_argument("--splits", type=int, default=10)
    parser.add_argument("--kstart", type=int, default=1)
    parser.add_argument("--kend", type=int, default=12)
    parser.add_argument("--Mrand", type=int, default=20)
    parser.add_argument("--refinement-steps", type=int, default=5)
    parser.add_argument("--workers", type=int, default=min(28, max(1, os.cpu_count() or 1)))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--target-test-error",
        type=float,
        default=0.1294,
        help="Stop once mean test L1 error is strictly below this value.",
    )
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    return parser.parse_args()


def numeric_value(row: Mapping[str, Any], keys: Sequence[str]) -> Optional[float]:
    lower_map = {key.lower(): key for key in row}
    for key in keys:
        actual_key = lower_map.get(key.lower())
        if actual_key is None:
            continue
        value = row.get(actual_key)
        if value in (None, ""):
            continue
        try:
            parsed = float(str(value).replace(",", ""))
        except ValueError:
            continue
        if np.isfinite(parsed):
            return parsed
    return None


def usable_record(row: Mapping[str, Any]) -> Optional[Tuple[float, float, float, float, float]]:
    heat_input = numeric_value(row, HEAT_INPUT_KEYS)
    so2 = numeric_value(row, SO2_KEYS)
    nox = numeric_value(row, NOX_KEYS)
    co2 = numeric_value(row, CO2_KEYS)
    nox_rate = numeric_value(row, NOX_RATE_KEYS)
    values = (heat_input, so2, nox, co2, nox_rate)
    if any(value is None or value <= 0 for value in values):
        return None
    return values  # type: ignore[return-value]


def usable_count(rows: Iterable[Mapping[str, Any]]) -> int:
    return sum(usable_record(row) is not None for row in rows)


def rows_from_csv(text: str) -> List[Dict[str, Any]]:
    return list(csv.DictReader(io.StringIO(text)))


def rows_from_json(text: str) -> List[Dict[str, Any]]:
    payload = json.loads(text)
    if isinstance(payload, list):
        return payload
    for key in ("items", "data", "results"):
        value = payload.get(key) if isinstance(payload, dict) else None
        if isinstance(value, list):
            return value
    raise ValueError("JSON data does not contain a list of rows.")


def read_rows(path: Path) -> List[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return rows_from_json(text)
    return rows_from_csv(text)


def quarter_date_range(year: int, quarter: int) -> Tuple[str, str]:
    if quarter == 1:
        return f"{year}-01-01", f"{year}-03-31"
    if quarter == 2:
        return f"{year}-04-01", f"{year}-06-30"
    if quarter == 3:
        return f"{year}-07-01", f"{year}-09-30"
    if quarter == 4:
        return f"{year}-10-01", f"{year}-12-31"
    raise ValueError("quarter must be between 1 and 4.")


def current_api_query(args: argparse.Namespace) -> List[Dict[str, Any]]:
    if args.aggregation in ("daily", "hourly"):
        begin_date, end_date = quarter_date_range(args.year, args.quarter)
        return [
            {
                "beginDate": begin_date,
                "endDate": end_date,
                "stateCode": args.state_code,
            }
        ]
    if args.aggregation == "monthly":
        first_month = 3 * (args.quarter - 1) + 1
        return [
            {"year": args.year, "month": month, "stateCode": args.state_code}
            for month in range(first_month, first_month + 3)
        ]
    return [{"year": args.year, "quarter": args.quarter, "stateCode": args.state_code}]


def fetch_url(url: str) -> Tuple[str, str]:
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=60) as response:
        content_type = response.headers.get("content-type", "")
        return response.read().decode("utf-8"), content_type


def fetch_legacy_endpoint(endpoint: str, args: argparse.Namespace) -> Tuple[str, str]:
    query = {"year": args.year, "quarter": args.quarter, "stateCode": args.state_code}
    if args.api_key:
        query["api_key"] = args.api_key
    url = endpoint + "?" + urllib.parse.urlencode(query)
    return fetch_url(url)


def extract_items(text: str) -> List[Dict[str, Any]]:
    payload = json.loads(text)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        items = payload.get("items")
        if isinstance(items, list):
            return items
    raise ValueError("EPA response did not contain rows.")


def fetch_current_endpoint(args: argparse.Namespace) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for base_query in current_api_query(args):
        page = 1
        while page <= args.max_pages:
            query = dict(base_query)
            query["page"] = page
            query["perPage"] = args.per_page
            if args.api_key:
                query["api_key"] = args.api_key
            url = (
                f"{EMISSIONS_BASE_URL}/{args.aggregation}?"
                + urllib.parse.urlencode(query)
            )
            text, _ = fetch_url(url)
            items = extract_items(text)
            rows.extend(items)
            if args.max_usable_rows > 0 and usable_count(rows) >= args.max_usable_rows:
                break
            if len(items) < args.per_page:
                break
            page += 1
        if page > args.max_pages:
            raise RuntimeError(f"Reached --max-pages={args.max_pages} before endpoint was exhausted.")
    return rows


def download_rows(args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], Path]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if args.aggregation != "quarterly":
        rows = fetch_current_endpoint(args)
        output_path = DATA_DIR / (
            f"campd_{args.aggregation}_{args.state_code.lower()}_q{args.quarter}_{args.year}.json"
        )
        output_path.write_text(json.dumps(rows), encoding="utf-8")
        return rows, output_path

    last_error: Optional[Exception] = None
    for endpoint in LEGACY_QUARTERLY_ENDPOINTS:
        try:
            text, content_type = fetch_legacy_endpoint(endpoint, args)
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            last_error = RuntimeError(f"{error.code} from {endpoint}: {detail}")
            continue
        except Exception as error:  # pragma: no cover - network/environment dependent
            last_error = error
            continue

        suffix = ".json" if "json" in content_type.lower() or text.lstrip().startswith(("{", "[")) else ".csv"
        output_path = DATA_DIR / f"campd_quarterly_{args.state_code.lower()}_q{args.quarter}_{args.year}{suffix}"
        output_path.write_text(text, encoding="utf-8")
        rows = rows_from_json(text) if suffix == ".json" else rows_from_csv(text)
        return rows, output_path

    message = "Could not download EPA CAMPD data."
    if last_error is not None:
        message += f" Last error: {last_error}"
    if not args.api_key:
        message += " EPA currently requires an API key; pass --api-key or set EPA_API_KEY."
    raise RuntimeError(message)


def build_design(
    rows: Iterable[Mapping[str, Any]],
    max_usable_rows: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    records = []
    for row in rows:
        values = usable_record(row)
        if values is None:
            continue
        records.append(values)

    if len(records) < 2:
        raise ValueError("No usable positive rows found for heat input/SO2/NOx/CO2/NOx rate.")

    data = np.asarray(records, dtype=float)
    if max_usable_rows > 0 and data.shape[0] > max_usable_rows:
        indices = rng.choice(data.shape[0], size=max_usable_rows, replace=False)
        data = data[np.sort(indices)]
    y = data[:, 0]
    X = np.log(data[:, 1:])

    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0, ddof=0)
    y_mean = y.mean()
    y_std = y.std(ddof=0)
    if np.any(X_std <= np.finfo(float).eps) or y_std <= np.finfo(float).eps:
        raise ValueError("Standardization failed because at least one column is constant.")

    return (X - X_mean) / X_std, (y - y_mean) / y_std


def predict_max_affine(X: np.ndarray, beta: np.ndarray, scale: float) -> np.ndarray:
    X_aug = np.column_stack((X, np.ones(X.shape[0])))
    return scale * np.max(X_aug @ beta.T, axis=1)


def run_one_split(
    X: np.ndarray,
    y: np.ndarray,
    k: int,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> Tuple[float, float]:
    permutation = rng.permutation(X.shape[0])
    needed = args.train_size + args.test_size
    train_indices = permutation[: args.train_size]
    test_indices = permutation[args.train_size:needed]

    beta, info = full_dimensional_random_search(
        X[train_indices],
        y[train_indices],
        k,
        args.Mrand,
        random_state=rng,
        loss="l1",
        refinement_steps=args.refinement_steps,
        workers=args.workers,
    )
    train_pred = predict_max_affine(X[train_indices], beta, 1.0)
    test_pred = predict_max_affine(X[test_indices], beta, 1.0)
    train_error = float(np.mean(np.abs(y[train_indices] - train_pred)))
    test_error = float(np.mean(np.abs(y[test_indices] - test_pred)))
    return train_error, test_error


def result_filename(args: argparse.Namespace) -> str:
    return (
        "epa_total_l1_nospectral"
        f"_k{args.kstart}-{args.kend}"
        f"_Mrand{args.Mrand}"
        f"_ref{args.refinement_steps}"
        f"_splits{args.splits}"
        f"_seed{args.seed}.npz"
    )


def validate_args(args: argparse.Namespace, n_rows: int, d: int) -> None:
    if args.train_size <= 0 or args.test_size <= 0:
        raise ValueError("train-size and test-size must be positive.")
    if args.train_size + args.test_size > n_rows:
        raise ValueError(
            f"Need {args.train_size + args.test_size} rows but only found {n_rows} usable rows."
        )
    if args.splits <= 0:
        raise ValueError("splits must be positive.")
    if args.kstart <= 0 or args.kend < args.kstart:
        raise ValueError("Require 1 <= kstart <= kend.")
    if args.Mrand <= 0:
        raise ValueError("Mrand must be positive.")
    if args.per_page <= 0 or args.per_page > 500:
        raise ValueError("per-page must be between 1 and 500.")
    if args.max_pages <= 0:
        raise ValueError("max-pages must be positive.")
    if args.max_usable_rows < 0:
        raise ValueError("max-usable-rows must be nonnegative.")
    if args.refinement_steps < 0:
        raise ValueError("refinement-steps must be nonnegative.")
    if args.workers <= 0:
        raise ValueError("workers must be positive.")
    if d <= 0:
        raise ValueError("The prepared design matrix must have at least one feature.")


def main() -> None:
    args = parse_args()
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
    validate_args(args, X.shape[0], X.shape[1])

    k_values = np.arange(args.kstart, args.kend + 1, dtype=int)
    train_errors = np.full((k_values.shape[0], args.splits), np.nan)
    test_errors = np.full((k_values.shape[0], args.splits), np.nan)
    best_k = None

    print(f"Using {X.shape[0]} usable rows from {data_path} with d={X.shape[1]}.")
    for k_index, k in enumerate(k_values):
        for split in range(args.splits):
            train_error, test_error = run_one_split(X, y, int(k), args, rng)
            train_errors[k_index, split] = train_error
            test_errors[k_index, split] = test_error
            print(
                f"k={k} split={split + 1}/{args.splits}: "
                f"train_l1={train_error:.4f}, test_l1={test_error:.4f}",
                flush=True,
            )

        mean_train = float(np.mean(train_errors[k_index]))
        mean_test = float(np.mean(test_errors[k_index]))
        print(f"k={k}: mean_train_l1={mean_train:.4f}, mean_test_l1={mean_test:.4f}")
        if mean_test < args.target_test_error:
            best_k = int(k)
            print(f"Stopping: k={k} beats target test error {args.target_test_error:.4f}.")
            break

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / result_filename(args)
    metadata: Dict[str, Any] = {
        "dataset": "EPA CAMPD California Q1 2019",
        "aggregation": args.aggregation,
        "data_path": str(data_path),
        "mode": "nospectral",
        "loss": "l1",
        "selection": "total_l1",
        "Mrand": args.Mrand,
        "refinement_steps": args.refinement_steps,
        "train_size": args.train_size,
        "test_size": args.test_size,
        "splits": args.splits,
        "seed": args.seed,
        "max_usable_rows": args.max_usable_rows,
        "target_test_error": args.target_test_error,
        "best_k_below_target": best_k,
        "defaults_source": "best options.md",
        "preprocessing": "positive rows only; log covariates; standardize X and y",
    }
    np.savez(
        str(output_path),
        k_values=k_values,
        train_errors=train_errors,
        test_errors=test_errors,
        metadata=json.dumps(metadata, indent=2, sort_keys=True),
    )
    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
