"""
evaluate.py
===========
Model evaluation and benchmarking for ML-IDS Zero Trust Cloud Research

Performance benchmarks from research paper (Table 5, §6.1):
  Algorithm     | Accuracy | Precision | Recall | F1    | FPR
  --------------|----------|-----------|--------|-------|-----
  Random Forest | 96.8%    | 97.1%     | 96.2%  | 96.6% | 2.1%
  SVM           | 94.2%    |  —        |  —     |  —    | 3.4%
  LSTM          | 98.1%    | 98.3%     | 97.9%  | 98.1% | 1.8%  ← BEST
  Autoencoder   | 91.5%    | 90.8%     | 92.3%  | 91.5% | 4.2%
  Iso. Forest   | 89.3%    | 88.5%     | 90.1%  | 89.3% | 5.1%
  XGBoost       | 97.3%    | 97.5%     | 97.1%  | 97.3% | 1.9%

Author: Dingaan Mahlatse Machethe — EC-Council University, ECCU500
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)


# ---------------------------------------------------------------------------
# Paper benchmark targets (Table 5) — used for validation assertions
# ---------------------------------------------------------------------------

PAPER_BENCHMARKS = {
    "random_forest": {"accuracy": 0.968, "fpr": 0.021, "precision": 0.971, "recall": 0.962, "f1": 0.966},
    "svm":           {"accuracy": 0.942, "fpr": 0.034},
    "lstm":          {"accuracy": 0.981, "fpr": 0.018, "precision": 0.983, "recall": 0.979, "f1": 0.981},
    "autoencoder":   {"accuracy": 0.915, "fpr": 0.042, "precision": 0.908, "recall": 0.923, "f1": 0.915},
    "xgboost":       {"accuracy": 0.973, "fpr": 0.019, "precision": 0.975, "recall": 0.971, "f1": 0.973},
}


def compute_fpr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute False Positive Rate = FP / (FP + TN).

    This is the key operational metric per paper §5.4:
    "The False Positive Rate (FPR) is additionally reported as it is the
    operational metric most directly relevant to SOC alert fatigue."
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return fp / (fp + tn) if (fp + tn) > 0 else 0.0


def evaluate_model(
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray = None,
) -> dict:
    """
    Compute full evaluation metrics for a single model.

    Returns a dict with accuracy, precision, recall, f1, fpr, auc (if probs provided).
    """
    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "fpr": compute_fpr(y_true, y_pred),
    }
    if y_prob is not None:
        try:
            metrics["auc_roc"] = roc_auc_score(y_true, y_prob)
        except ValueError:
            metrics["auc_roc"] = None

    return metrics


def compare_models(results: list[dict]) -> pd.DataFrame:
    """
    Build a comparison DataFrame from a list of evaluate_model() results.

    Includes benchmark deltas vs paper Table 5 targets.
    """
    df = pd.DataFrame(results)

    # Add delta vs paper benchmarks
    accuracy_deltas = []
    fpr_deltas = []
    for _, row in df.iterrows():
        name = row["model"].lower().replace(" ", "_").replace("-", "_")
        bench = PAPER_BENCHMARKS.get(name, {})
        if bench:
            accuracy_deltas.append(round(row["accuracy"] - bench.get("accuracy", row["accuracy"]), 4))
            fpr_deltas.append(round(row["fpr"] - bench.get("fpr", row["fpr"]), 4))
        else:
            accuracy_deltas.append(None)
            fpr_deltas.append(None)

    df["accuracy_vs_paper"] = accuracy_deltas
    df["fpr_vs_paper"] = fpr_deltas

    # Format as percentages for readability
    pct_cols = ["accuracy", "precision", "recall", "f1", "fpr"]
    for col in pct_cols:
        if col in df.columns:
            df[col] = (df[col] * 100).round(1)

    df = df.sort_values("accuracy", ascending=False).reset_index(drop=True)
    return df


def print_comparison_table(df: pd.DataFrame) -> None:
    """Pretty-print the model comparison table matching paper Table 5 format."""
    print("\n" + "=" * 80)
    print("  ML Algorithm Performance Comparison (NSL-KDD Dataset)")
    print("  Ref: Table 5, §6.1 — Machethe (2026), EC-Council University")
    print("=" * 80)
    display_cols = [c for c in ["model", "accuracy", "precision", "recall", "f1", "fpr", "auc_roc"] if c in df.columns]
    print(df[display_cols].to_string(index=False))
    print("=" * 80)
    print("  Detection Rate = Recall. All metrics from 10-fold stratified CV.")
    print("  FPR = False Positive Rate (key SOC operational metric).")
    print("  Best performer: LSTM — 98.1% accuracy, 1.8% FPR (paper benchmark)")
    print("=" * 80 + "\n")


def validate_against_paper(model_name: str, metrics: dict, tolerance: float = 0.02) -> bool:
    """
    Assert that computed metrics are within tolerance of paper benchmarks.

    tolerance: allowable deviation (default ±2 percentage points)
    Returns True if within tolerance, False otherwise.
    """
    bench = PAPER_BENCHMARKS.get(model_name.lower().replace(" ", "_"), None)
    if bench is None:
        print(f"[WARNING] No benchmark found for model '{model_name}'. Skipping validation.")
        return True

    passed = True
    for metric_name, target in bench.items():
        actual = metrics.get(metric_name)
        if actual is None:
            continue
        delta = abs(actual - target)
        status = "PASS" if delta <= tolerance else "FAIL"
        if status == "FAIL":
            passed = False
        print(
            f"  [{status}] {model_name} {metric_name}: "
            f"actual={actual:.3f}, paper={target:.3f}, delta={delta:.3f}"
        )
    return passed
