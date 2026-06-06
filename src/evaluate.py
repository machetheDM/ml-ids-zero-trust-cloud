"""
evaluate.py
===========
Model evaluation, metrics computation, and CSV reporting for ML-IDS Zero Trust Cloud

Paper benchmarks (Table 5, §6.1):
  Algorithm     | Accuracy | Precision | Recall | F1-Score | FPR
  --------------|----------|-----------|--------|----------|-----
  LSTM          | 98.1%    | 98.3%     | 97.9%  | 98.1%    | 1.8%  <- BEST
  XGBoost       | 97.3%    | 97.5%     | 97.1%  | 97.3%    | 1.9%
  Random Forest | 96.8%    | 97.1%     | 96.2%  | 96.6%    | 2.1%
  SVM           | 94.2%    |  -        |  -     |  -       | 3.4%
  Autoencoder   | 91.5%    | 90.8%     | 92.3%  | 91.5%    | 4.2%

Outputs:
  - results/metrics.csv  — per-model metrics including training time and latency
  - Console comparison table

Author: Dingaan Mahlatse Machethe — EC-Council University, ECCU500
"""

import os
import time
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
METRICS_CSV = os.path.join(RESULTS_DIR, "metrics.csv")

# ---------------------------------------------------------------------------
# Paper benchmark targets — source of truth (Table 5, §6.1)
# ---------------------------------------------------------------------------

PAPER_BENCHMARKS = {
    "LSTM":          {"accuracy": 0.981, "fpr": 0.018, "precision": 0.983, "recall": 0.979, "f1": 0.981},
    "XGBoost":       {"accuracy": 0.973, "fpr": 0.019, "precision": 0.975, "recall": 0.971, "f1": 0.973},
    "Random Forest": {"accuracy": 0.968, "fpr": 0.021, "precision": 0.971, "recall": 0.962, "f1": 0.966},
    "SVM":           {"accuracy": 0.942, "fpr": 0.034},
    "Autoencoder":   {"accuracy": 0.915, "fpr": 0.042, "precision": 0.908, "recall": 0.923, "f1": 0.915},
}


# ---------------------------------------------------------------------------
# Core metric functions
# ---------------------------------------------------------------------------

def compute_fpr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    False Positive Rate = FP / (FP + TN).

    Paper §5.4: "FPR is the operational metric most directly relevant to
    SOC alert fatigue in production environments."
    Collapses multiclass labels to binary before computing.
    """
    y_b = (y_true > 0).astype(int)
    y_p = (y_pred > 0).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_b, y_p, labels=[0, 1]).ravel()
    return fp / (fp + tn) if (fp + tn) > 0 else 0.0


def measure_latency(predict_fn, X: np.ndarray, n_repeats: int = 3) -> float:
    """
    Measure inference latency in milliseconds per sample.

    Runs predict_fn n_repeats times; reports best (minimum) time.
    """
    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        predict_fn(X)
        times.append(time.perf_counter() - t0)
    return (min(times) / len(X)) * 1000.0


def evaluate_model(
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray = None,
    training_time: float = 0.0,
    latency_ms: float = 0.0,
) -> dict:
    """
    Compute full evaluation metrics for one model.

    Returns a dict compatible with evaluate_all() and validate_against_paper().
    All sklearn metrics use average='weighted' to match paper §5.4 methodology.
    """
    y_b = (y_true > 0).astype(int)
    y_p = (y_pred > 0).astype(int)

    metrics = {
        "model":             model_name,
        "accuracy":          round(accuracy_score(y_b, y_p), 4),
        "precision":         round(precision_score(y_b, y_p, average="weighted", zero_division=0), 4),
        "recall":            round(recall_score(y_b, y_p, average="weighted", zero_division=0), 4),
        "f1":                round(f1_score(y_b, y_p, average="weighted", zero_division=0), 4),
        "fpr":               round(compute_fpr(y_true, y_pred), 4),
        "training_time_s":   round(training_time, 2),
        "latency_ms_sample": round(latency_ms, 4),
    }

    if y_prob is not None:
        try:
            p = y_prob[:, 1] if (y_prob.ndim == 2) else y_prob.flatten()
            metrics["auc_roc"] = round(roc_auc_score(y_b, p), 4)
        except Exception:
            metrics["auc_roc"] = None
    else:
        metrics["auc_roc"] = None

    return metrics


# ---------------------------------------------------------------------------
# Aggregation and CSV export
# ---------------------------------------------------------------------------

def evaluate_all(results: list) -> pd.DataFrame:
    """
    Build comparison DataFrame from a list of evaluate_model() result dicts.
    Saves to results/metrics.csv.
    """
    df = pd.DataFrame(results)
    for col in ["accuracy", "precision", "recall", "f1", "fpr"]:
        if col in df.columns:
            df[f"{col}_pct"] = (df[col] * 100).round(1)

    df = df.sort_values("accuracy", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))

    os.makedirs(RESULTS_DIR, exist_ok=True)
    df.to_csv(METRICS_CSV, index=False)
    print(f"Metrics saved to: {METRICS_CSV}")
    return df


def compare_models(results: list) -> pd.DataFrame:
    """Alias for evaluate_all() — backwards compatible."""
    return evaluate_all(results)


def print_comparison_table(df: pd.DataFrame) -> None:
    """Pretty-print the model comparison table matching paper Table 5 format."""
    rename = {
        "accuracy_pct": "Acc%", "precision_pct": "Prec%",
        "recall_pct": "Rec%", "f1_pct": "F1%", "fpr_pct": "FPR%",
        "training_time_s": "Train(s)", "latency_ms_sample": "Lat(ms)",
    }
    cols = [c for c in
            ["rank", "model", "accuracy_pct", "precision_pct",
             "recall_pct", "f1_pct", "fpr_pct",
             "training_time_s", "latency_ms_sample", "auc_roc"]
            if c in df.columns]
    print("\n" + "=" * 90)
    print("  ML Algorithm Performance — NSL-KDD (Machethe 2026, Table 5, ECCU500)")
    print("=" * 90)
    print(df[cols].rename(columns=rename).to_string(index=False))
    print("=" * 90)
    print("  Recall = Detection Rate. FPR = key SOC operational metric.")
    print("  Best: LSTM — 98.1% accuracy, 1.8% FPR (paper §6.2.1)")
    print("=" * 90 + "\n")


# ---------------------------------------------------------------------------
# Paper benchmark validation
# ---------------------------------------------------------------------------

def validate_against_paper(model_name: str, metrics: dict, tolerance: float = 0.02) -> bool:
    """
    Assert computed metrics are within tolerance of paper Table 5 benchmarks.

    Default tolerance: ±2 percentage points (paper §6.1 notes experimental variance).
    Returns True if all measured metrics pass, False otherwise.
    """
    bench = PAPER_BENCHMARKS.get(model_name)
    if not bench:
        print(f"[WARNING] No benchmark found for '{model_name}'.")
        return True

    passed = True
    for k, target in bench.items():
        actual = metrics.get(k)
        if actual is None:
            continue
        delta = abs(actual - target)
        tag = "PASS" if delta <= tolerance else "FAIL"
        if tag == "FAIL":
            passed = False
        print(f"  [{tag}] {model_name} {k}: actual={actual:.3f}, paper={target:.3f}, delta={delta:.4f}")
    return passed
