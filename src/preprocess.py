"""
preprocess.py
=============
NSL-KDD Data Pipeline for ML-IDS Zero Trust Cloud Research

Paper: "Machine Learning-Based Intrusion Detection for Cloud Network Security:
        A Zero Trust Architecture Approach"
Author: Dingaan Mahlatse Machethe
Institution: EC-Council University — ECCU500: Managing Secure Network Systems

Pipeline steps (per paper §5.2):
  1. Download NSL-KDD training set from GitHub
  2. Assign the 42 standard NSL-KDD column names
  3. One-hot encode categorical features: protocol_type, service, flag
  4. Apply MinMaxScaler normalisation to all numerical features
  5. Apply SMOTE oversampling to handle class imbalance
  6. Apply RFECV (RandomForestClassifier) to select top 25 features
  7. Perform stratified 80/20 train-test split
  8. Save processed arrays to data/ as .npy files
  9. Save fitted scaler and feature selector to results/models/ as .pkl files
"""

import os
import io
import logging
import requests
import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — all values sourced directly from the research paper
# ---------------------------------------------------------------------------

NSL_KDD_URL = (
    "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain+.txt"
)

# 42 standard NSL-KDD column names (41 features + label)
NSL_KDD_COLUMNS = [
    "duration", "protocol_type", "service", "flag",
    "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent", "hot",
    "num_failed_logins", "logged_in", "num_compromised", "root_shell",
    "su_attempted", "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
    "label",
]

# Categorical columns to one-hot encode (per paper §5.2)
CATEGORICAL_COLS = ["protocol_type", "service", "flag"]

# Attack category mapping — 4 categories per paper (DoS, Probe, R2L, U2R)
# Binary classification: 0 = normal, 1 = attack
ATTACK_LABEL_MAP = {
    "normal": 0,
    # DoS
    "back": 1, "land": 1, "neptune": 1, "pod": 1, "smurf": 1,
    "teardrop": 1, "apache2": 1, "udpstorm": 1, "processtable": 1,
    "worm": 1, "mailbomb": 1,
    # Probe
    "ipsweep": 1, "nmap": 1, "portsweep": 1, "satan": 1,
    "mscan": 1, "saint": 1,
    # R2L
    "ftp_write": 1, "guess_passwd": 1, "imap": 1, "multihop": 1,
    "phf": 1, "spy": 1, "warezclient": 1, "warezmaster": 1,
    "sendmail": 1, "named": 1, "snmpgetattack": 1, "snmpguess": 1,
    "xlock": 1, "xsnoop": 1, "httptunnel": 1,
    # U2R
    "buffer_overflow": 1, "loadmodule": 1, "perl": 1, "rootkit": 1,
    "ps": 1, "sqlattack": 1, "xterm": 1,
}

# Feature selection target — paper §5.2: "reduce to the 25 most informative features"
N_FEATURES_TO_SELECT = 25

# Train-test split — paper §5.2: "stratified 80/20 train-test split"
TEST_SIZE = 0.20
RANDOM_STATE = 42

# RFECV cross-validation folds
RFECV_CV = 5

# Paths
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")
MODELS_DIR = os.path.join(ROOT_DIR, "results", "models")


# ---------------------------------------------------------------------------
# Step 1: Download NSL-KDD dataset
# ---------------------------------------------------------------------------

def download_nsl_kdd(url: str = NSL_KDD_URL) -> pd.DataFrame:
    """Download NSL-KDD training set and assign the 42 standard column names."""
    logger.info("Downloading NSL-KDD dataset from %s", url)
    response = requests.get(url, timeout=60)
    response.raise_for_status()

    # The raw file has no header; columns include an optional 43rd difficulty col
    raw = pd.read_csv(
        io.StringIO(response.text),
        header=None,
        names=NSL_KDD_COLUMNS + ["difficulty"],  # handle optional 43rd col
    )

    # Drop the difficulty score column — not part of the 42 standard columns
    if "difficulty" in raw.columns:
        raw = raw.drop(columns=["difficulty"])

    logger.info(
        "Downloaded %d records with %d columns.", len(raw), len(raw.columns)
    )
    return raw


# ---------------------------------------------------------------------------
# Step 2: Label encoding (binary: normal=0, attack=1)
# ---------------------------------------------------------------------------

def encode_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Convert multi-class attack labels to binary (0=normal, 1=attack)."""
    df = df.copy()
    df["label"] = df["label"].str.strip().str.lower()
    df["label"] = df["label"].map(ATTACK_LABEL_MAP)

    unknown = df["label"].isna().sum()
    if unknown > 0:
        logger.warning("%d rows had unrecognised labels — marking as attack.", unknown)
        df["label"] = df["label"].fillna(1).astype(int)
    else:
        df["label"] = df["label"].astype(int)

    normal_count = (df["label"] == 0).sum()
    attack_count = (df["label"] == 1).sum()
    ratio = normal_count / len(df) * 100
    logger.info(
        "Label distribution — Normal: %d (%.1f%%), Attack: %d (%.1f%%)",
        normal_count, ratio, attack_count, 100 - ratio,
    )
    return df


# ---------------------------------------------------------------------------
# Step 3: One-hot encode categorical features
# ---------------------------------------------------------------------------

def one_hot_encode(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode protocol_type, service, and flag columns."""
    logger.info("One-hot encoding categorical features: %s", CATEGORICAL_COLS)
    df = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=False)
    logger.info("Feature count after OHE: %d", df.shape[1] - 1)  # minus label
    return df


# ---------------------------------------------------------------------------
# Step 4: MinMaxScaler normalisation
# ---------------------------------------------------------------------------

def normalise_features(
    X: np.ndarray,
    scaler: MinMaxScaler = None,
    fit: bool = True,
) -> tuple[np.ndarray, MinMaxScaler]:
    """Apply MinMaxScaler normalisation to all numerical features."""
    if scaler is None:
        scaler = MinMaxScaler()
    if fit:
        logger.info("Fitting MinMaxScaler on training data.")
        X_scaled = scaler.fit_transform(X)
    else:
        logger.info("Applying pre-fitted MinMaxScaler to test data.")
        X_scaled = scaler.transform(X)
    return X_scaled, scaler


# ---------------------------------------------------------------------------
# Step 5: SMOTE oversampling
# ---------------------------------------------------------------------------

def apply_smote(
    X: np.ndarray,
    y: np.ndarray,
    random_state: int = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply SMOTE to handle class imbalance (paper §5.2)."""
    logger.info(
        "Applying SMOTE — pre-resampling shape: X=%s, class dist=%s",
        X.shape,
        dict(zip(*np.unique(y, return_counts=True))),
    )
    smote = SMOTE(random_state=random_state)
    X_res, y_res = smote.fit_resample(X, y)
    logger.info(
        "Post-SMOTE shape: X=%s, class dist=%s",
        X_res.shape,
        dict(zip(*np.unique(y_res, return_counts=True))),
    )
    return X_res, y_res


# ---------------------------------------------------------------------------
# Step 6: RFECV feature selection — top 25 features
# ---------------------------------------------------------------------------

def select_features_rfecv(
    X: np.ndarray,
    y: np.ndarray,
    n_features: int = N_FEATURES_TO_SELECT,
    cv: int = RFECV_CV,
    random_state: int = RANDOM_STATE,
) -> tuple[np.ndarray, RFECV]:
    """
    Recursive Feature Elimination with Cross-Validation (RFECV).

    Uses RandomForestClassifier as the estimator.
    Target: top 25 features (paper §5.2: "reduce the 41-dimensional NSL-KDD
    feature space to the 25 most informative features").
    """
    logger.info(
        "Running RFECV (RandomForestClassifier, cv=%d) to select top %d features...",
        cv, n_features,
    )
    estimator = RandomForestClassifier(
        n_estimators=100,
        criterion="gini",
        n_jobs=-1,
        random_state=random_state,
    )
    selector = RFECV(
        estimator=estimator,
        min_features_to_select=n_features,
        cv=cv,
        scoring="f1_weighted",
        n_jobs=-1,
        verbose=0,
    )
    selector.fit(X, y)

    actual_selected = selector.n_features_
    logger.info("RFECV selected %d features (target: %d).", actual_selected, n_features)

    # If RFECV selected more than n_features, use feature importances to trim
    if actual_selected > n_features:
        logger.info(
            "Trimming to top %d by feature importance.", n_features
        )
        importances = selector.estimator_.feature_importances_
        selected_indices = np.where(selector.support_)[0]
        top_n_local = np.argsort(importances)[::-1][:n_features]
        final_mask = np.zeros(X.shape[1], dtype=bool)
        final_mask[selected_indices[top_n_local]] = True
        selector.support_ = final_mask

    X_selected = selector.transform(X)
    logger.info("Feature matrix shape after selection: %s", X_selected.shape)
    return X_selected, selector


# ---------------------------------------------------------------------------
# Step 7: Stratified 80/20 train-test split
# ---------------------------------------------------------------------------

def stratified_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stratified 80/20 train-test split preserving class distribution."""
    logger.info(
        "Applying stratified %.0f/%.0f train-test split.",
        (1 - test_size) * 100, test_size * 100,
    )
    sss = StratifiedShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    train_idx, test_idx = next(sss.split(X, y))
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    logger.info(
        "Train: %d samples | Test: %d samples", len(X_train), len(X_test)
    )
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Steps 8–9: Save outputs
# ---------------------------------------------------------------------------

def save_arrays(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    output_dir: str = DATA_DIR,
) -> None:
    """Save processed NumPy arrays to the data/ folder as .npy files."""
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "X_train.npy"), X_train)
    np.save(os.path.join(output_dir, "X_test.npy"), X_test)
    np.save(os.path.join(output_dir, "y_train.npy"), y_train)
    np.save(os.path.join(output_dir, "y_test.npy"), y_test)
    logger.info("Saved processed arrays to %s", output_dir)


def save_artifacts(
    scaler: MinMaxScaler,
    selector: RFECV,
    feature_names: list[str],
    output_dir: str = MODELS_DIR,
) -> None:
    """Save fitted scaler and feature selector as .pkl files."""
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(output_dir, "scaler.pkl"))
    joblib.dump(selector, os.path.join(output_dir, "rfecv_selector.pkl"))

    # Also save selected feature names for interpretability
    selected_mask = selector.support_
    selected_features = [f for f, m in zip(feature_names, selected_mask) if m]
    pd.Series(selected_features).to_csv(
        os.path.join(output_dir, "selected_features.csv"), index=False, header=["feature"]
    )
    logger.info(
        "Saved scaler.pkl, rfecv_selector.pkl, selected_features.csv to %s", output_dir
    )
    logger.info("Selected features: %s", selected_features)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline() -> dict:
    """
    Execute the full NSL-KDD preprocessing pipeline.

    Returns a dict with shapes and class distributions for verification.
    """
    logger.info("=" * 60)
    logger.info("ML-IDS Zero Trust Cloud — NSL-KDD Preprocessing Pipeline")
    logger.info("Paper: ECCU500 Managing Secure Network Systems")
    logger.info("Author: Dingaan Mahlatse Machethe")
    logger.info("=" * 60)

    # Step 1: Download
    df = download_nsl_kdd()

    # Step 2: Encode labels to binary
    df = encode_labels(df)

    # Step 3: One-hot encode categorical features
    df = one_hot_encode(df)

    # Separate features and labels
    y = df["label"].values
    X_df = df.drop(columns=["label"])
    feature_names = list(X_df.columns)
    X = X_df.values.astype(np.float32)

    logger.info("Feature matrix shape before normalisation: %s", X.shape)

    # Step 4: MinMaxScaler normalisation (fit on full data before SMOTE/split)
    # Note: In a strict pipeline, fit only on training split. Here we fit on
    # full data first to enable RFECV, then re-fit on training split only.
    X_scaled, scaler_full = normalise_features(X, fit=True)

    # Step 5: SMOTE — applied before split to generate balanced training data
    X_resampled, y_resampled = apply_smote(X_scaled, y)

    # Step 6: RFECV feature selection — run on SMOTE-balanced full set
    X_selected, selector = select_features_rfecv(X_resampled, y_resampled)

    # Step 7: Stratified 80/20 split on SMOTE+RFECV processed data
    X_train_raw, X_test_raw, y_train, y_test = stratified_split(
        X_selected, y_resampled
    )

    # Re-fit scaler on training split only (best practice)
    X_train, scaler = normalise_features(X_train_raw, fit=True)
    X_test, _ = normalise_features(X_test_raw, scaler=scaler, fit=False)

    # Step 8: Save arrays
    save_arrays(X_train, X_test, y_train, y_test)

    # Step 9: Save scaler and selector
    # Build feature names post-OHE for the selector mask
    save_artifacts(scaler, selector, feature_names)

    summary = {
        "X_train_shape": X_train.shape,
        "X_test_shape": X_test.shape,
        "y_train_shape": y_train.shape,
        "y_test_shape": y_test.shape,
        "n_features_selected": int(selector.support_.sum()),
        "train_class_dist": dict(zip(*np.unique(y_train, return_counts=True))),
        "test_class_dist": dict(zip(*np.unique(y_test, return_counts=True))),
    }

    logger.info("=" * 60)
    logger.info("Pipeline complete. Summary:")
    for k, v in summary.items():
        logger.info("  %-25s: %s", k, v)
    logger.info("=" * 60)

    return summary


if __name__ == "__main__":
    run_pipeline()
