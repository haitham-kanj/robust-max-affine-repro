from pathlib import Path
import numpy as np

BASE = Path("results/figure2")

def summarize(label, train, test):
    print(f"{label:10s} train {train.mean():.4f} +/- {train.std(ddof=1):.4f} | test {test.mean():.4f} +/- {test.std(ddof=1):.4f}")

paper = np.load(BASE / "epa_paper_methods_daily_splits10_seed1.npz", allow_pickle=True)
for method, train, test in zip(paper["methods"], paper["train_errors"], paper["test_errors"]):
    summarize(str(method), train, test)

ram = np.load(BASE / "epa_total_l1_nospectral_k4-5_Mrand20_ref5_splits10_seed1.npz", allow_pickle=True)
for k, train, test in zip(ram["k_values"], ram["train_errors"], ram["test_errors"]):
    summarize(f"RAM k={int(k)}", train, test)
