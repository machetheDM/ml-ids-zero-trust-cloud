"""
drift_monitor.py
================
Data & Model Drift Monitoring for ML-IDS Zero Trust Cloud.

Paper: "Machine Learning-Based Intrusion Detection for Cloud Network Security:
        A Zero Trust Architecture Approach"
Author: Dingaan Mahlatse Machethe

Monitors:
  1. Data Drift — PSI (Population Stability Index) per feature,
     comparing reference (training) vs current (production) distributions.
  2. Prediction Drift — monitors prediction distribution shifts over time.
  3. Model Performance Degradation — if ground-truth labels become available
     (e.g., from SOC analyst feedback), compares current vs reference metrics.

Uses Evidently AI for drift reports and statistical tests.

Usage:
  python src/drift_monitor.py                          # single drift report
  python src/drift_monitor.py --schedule               # simulate 5 time windows
  python src/drift_monitor.py --output-dir reports/     # custom output dir
"""

import os
import sys
import json
import argparse
import logging
import datetime
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")
REPORTS_DIR = os.path.join(ROOT_DIR, "results", "drift_reports")
MODELS_DIR = os.path.join(ROOT_DIR, "results", "models")

# Drift thresholds
PSI_WARNING_THRESHOLD = 0.10   # PSI > 0.10 = moderate drift
PSI_CRITICAL_THRESHOLD = 0.25  # PSI > 0.25 = severe drift
DRIFT_SHARE_WARNING = 0.30     # >30% features drifted = warning
DRIFT_SHARE_CRITICAL = 0.50    # >50% features drifted = critical


def load_data():
    """Load preprocessed train/test arrays."""
    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))
    return X_train, X_test, y_train, y_test


def load_feature_names():
    """Load selected feature names from pipeline output."""
    path = os.path.join(MODELS_DIR, "selected_features.csv")
    if os.path.exists(path):
        names = pd.read_csv(path)["feature"].tolist()
    else:
        names = [f"feature_{i}" for i in range(25)]
    return names


def compute_psi(expected: np.ndarray, actual: np.ndarray,
                bins: int = 10) -> float:
    """
    Population Stability Index per feature.

    PSI = Σ (actual_i - expected_i) * ln(actual_i / expected_i)

    Interpretation:
      PSI < 0.10  → no significant drift
      0.10–0.25   → moderate drift (warning)
      PSI > 0.25  → severe drift (critical)
    """
    expected = np.asarray(expected, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)

    # Use shared bin edges from expected distribution
    bin_edges = np.percentile(expected, np.linspace(0, 100, bins + 1))
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) < 2:
        return 0.0

    expected_hist, _ = np.histogram(expected, bins=bin_edges, density=False)
    actual_hist, _ = np.histogram(actual, bins=bin_edges, density=False)

    # Normalise to proportions
    expected_prop = expected_hist / (expected_hist.sum() + 1e-9)
    actual_prop = actual_hist / (actual_hist.sum() + 1e-9)

    # Clip to avoid log(0)
    expected_prop = np.clip(expected_prop, 1e-9, 1)
    actual_prop = np.clip(actual_prop, 1e-9, 1)

    psi = np.sum((actual_prop - expected_prop) * np.log(actual_prop / expected_prop))
    return float(psi)


def compute_feature_drift(reference: np.ndarray, current: np.ndarray,
                          feature_names: list[str]) -> pd.DataFrame:
    """Compute PSI for every feature. Returns sorted DataFrame."""
    n_features = reference.shape[1]
    rows = []
    for i in range(n_features):
        psi = compute_psi(reference[:, i], current[:, i])
        status = (
            "CRITICAL" if psi > PSI_CRITICAL_THRESHOLD
            else "WARNING" if psi > PSI_WARNING_THRESHOLD
            else "STABLE"
        )
        rows.append({
            "feature": feature_names[i] if i < len(feature_names) else f"f{i}",
            "psi": round(psi, 6),
            "status": status,
        })
    df = pd.DataFrame(rows).sort_values("psi", ascending=False)
    return df


def compute_prediction_drift(y_ref_pred: np.ndarray,
                              y_curr_pred: np.ndarray) -> dict:
    """Compare prediction distribution between reference and current."""
    ref_normal = float((y_ref_pred == 0).mean())
    curr_normal = float((y_curr_pred == 0).mean())
    ref_attack = float((y_ref_pred == 1).mean())
    curr_attack = float((y_curr_pred == 1).mean())

    delta_normal = curr_normal - ref_normal
    delta_attack = curr_attack - ref_attack

    alert = abs(delta_attack) > 0.10  # >10pp shift in attack rate

    return {
        "reference": {"normal_rate": round(ref_normal, 4),
                       "attack_rate": round(ref_attack, 4)},
        "current": {"normal_rate": round(curr_normal, 4),
                     "attack_rate": round(curr_attack, 4)},
        "delta_attack_rate": round(delta_attack, 4),
        "alert": alert,
        "message": (
            f"Attack prediction rate shifted by {delta_attack:+.1%}. "
            "Investigate for concept drift or new attack patterns."
            if alert else "Prediction distribution stable."
        ),
    }


def generate_drift_report(
    X_train: np.ndarray,
    X_current: np.ndarray,
    y_train: np.ndarray = None,
    y_current: np.ndarray = None,
    y_train_pred: np.ndarray = None,
    y_current_pred: np.ndarray = None,
    feature_names: list[str] = None,
    window_label: str = "current",
    output_dir: str = REPORTS_DIR,
) -> dict:
    """
    Generate a comprehensive drift report.

    Returns a dict with:
      - feature_drift: per-feature PSI values
      - summary: drift share, status, top drifted features
      - prediction_drift: if predictions provided
      - performance_drift: if ground-truth labels provided
    """
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. Feature drift
    logger.info("Computing PSI for %d features...", X_train.shape[1])
    feature_drift_df = compute_feature_drift(X_train, X_current, feature_names)
    n_drifted = int((feature_drift_df["status"] != "STABLE").sum())
    drift_share = n_drifted / len(feature_drift_df)

    if drift_share > DRIFT_SHARE_CRITICAL:
        overall_status = "CRITICAL"
    elif drift_share > DRIFT_SHARE_WARNING:
        overall_status = "WARNING"
    else:
        overall_status = "STABLE"

    top_drifted = feature_drift_df.head(5).to_dict(orient="records")

    report = {
        "timestamp": timestamp,
        "window": window_label,
        "n_features": int(X_train.shape[1]),
        "n_reference_samples": int(len(X_train)),
        "n_current_samples": int(len(X_current)),
        "feature_drift": {
            "n_drifted": n_drifted,
            "drift_share": round(drift_share, 4),
            "overall_status": overall_status,
            "top_drifted": top_drifted,
            "all_features": feature_drift_df.to_dict(orient="records"),
        },
    }

    # 2. Prediction drift
    if y_train_pred is not None and y_current_pred is not None:
        report["prediction_drift"] = compute_prediction_drift(
            y_train_pred, y_current_pred
        )

    # 3. Performance drift (if labels available)
    if y_train is not None and y_current is not None:
        from evaluate import compute_all_metrics
        ref_metrics = compute_all_metrics(y_train, y_train_pred, None)
        curr_metrics = compute_all_metrics(y_current, y_current_pred, None)
        report["performance_drift"] = {
            "reference": {k: round(v, 4) for k, v in ref_metrics.items()},
            "current": {k: round(v, 4) for k, v in curr_metrics.items()},
            "accuracy_delta": round(
                curr_metrics["accuracy"] - ref_metrics["accuracy"], 4
            ),
            "f1_delta": round(
                curr_metrics["f1"] - ref_metrics["f1"], 4
            ),
        }

    # Save report
    report_path = os.path.join(
        output_dir, f"drift_report_{window_label}_{timestamp}.json"
    )
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    # Save feature drift CSV
    csv_path = os.path.join(
        output_dir, f"feature_drift_{window_label}_{timestamp}.csv"
    )
    feature_drift_df.to_csv(csv_path, index=False)

    logger.info("Drift report saved to %s", report_path)
    logger.info(
        "Overall: %s | Drifted features: %d/%d (%.1f%%)",
        overall_status, n_drifted, len(feature_drift_df),
        drift_share * 100,
    )

    return report


def simulate_time_windows(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    n_windows: int = 5,
    output_dir: str = REPORTS_DIR,
) -> list[dict]:
    """
    Simulate drift across multiple time windows by injecting
    progressively more noise into the test data.

    This demonstrates how drift monitoring would work in production
    where data arrives in batches over time.
    """
    feature_names = load_feature_names()
    reports = []

    # Load best model for predictions
    from models import RandomForestIDS
    model = RandomForestIDS()
    model_path = os.path.join(MODELS_DIR, "random_forest.pkl")
    if os.path.exists(model_path):
        model = RandomForestIDS.load(name="random_forest")
    else:
        model.train(X_train, y_train)

    y_train_pred = model.predict(X_train)

    for w in range(n_windows):
        noise_level = 0.02 * (w + 1)  # 2%, 4%, 6%, 8%, 10%
        noise = np.random.normal(0, noise_level, X_test.shape).astype(np.float32)
        X_window = X_test + noise

        y_window_pred = model.predict(X_window)

        logger.info("--- Window %d/%d (noise=%.0f%%) ---",
                     w + 1, n_windows, noise_level * 100)

        report = generate_drift_report(
            X_train=X_train,
            X_current=X_window,
            y_train=y_train,
            y_current=y_test,
            y_train_pred=y_train_pred,
            y_current_pred=y_window_pred,
            feature_names=feature_names,
            window_label=f"window_{w+1}",
            output_dir=output_dir,
        )
        reports.append(report)

    # Save summary across all windows
    summary = {
        "generated_at": datetime.datetime.now().isoformat(),
        "n_windows": n_windows,
        "windows": [
            {
                "window": r["window"],
                "status": r["feature_drift"]["overall_status"],
                "drift_share": r["feature_drift"]["drift_share"],
                "top_feature": r["feature_drift"]["top_drifted"][0]["feature"]
                if r["feature_drift"]["top_drifted"] else None,
            }
            for r in reports
        ],
    }
    summary_path = os.path.join(output_dir, "drift_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Drift summary saved to %s", summary_path)
    return reports


def main():
    parser = argparse.ArgumentParser(
        description="ML-IDS Drift Monitor"
    )
    parser.add_argument(
        "--schedule", action="store_true",
        help="Simulate drift across 5 time windows",
    )
    parser.add_argument(
        "--output-dir", type=str, default=REPORTS_DIR,
        help="Output directory for drift reports",
    )
    args = parser.parse_args()

    X_train, X_test, y_train, y_test = load_data()
    feature_names = load_feature_names()

    if args.schedule:
        simulate_time_windows(
            X_train, X_test, y_train, y_test,
            n_windows=5, output_dir=args.output_dir,
        )
    else:
        # Single report: training vs test
        from models import RandomForestIDS
        model = RandomForestIDS()
        model_path = os.path.join(MODELS_DIR, "random_forest.pkl")
        if os.path.exists(model_path):
            model = RandomForestIDS.load(name="random_forest")
        else:
            model.train(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        generate_drift_report(
            X_train=X_train,
            X_current=X_test,
            y_train=y_train,
            y_current=y_test,
            y_train_pred=y_train_pred,
            y_current_pred=y_test_pred,
            feature_names=feature_names,
            window_label="test_set",
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
