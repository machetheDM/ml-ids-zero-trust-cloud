"""
train_mlflow.py
==============
MLflow-tracked training pipeline for all 5 IDS models.

Paper: "Machine Learning-Based Intrusion Detection for Cloud Network Security:
        A Zero Trust Architecture Approach"
Author: Dingaan Mahlatse Machethe

Adds MLOps layer on top of models.py:
  - Experiment tracking (params, metrics, artifacts)
  - Model registry with versioning
  - Automatic best-model promotion
  - Training run comparison

Usage:
  python src/train_mlflow.py                # train all 5 models
  python src/train_mlflow.py --model lstm    # train single model
  mlflow ui --port 5000                      # launch tracking UI
"""

import os
import sys
import time
import argparse
import logging
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.keras
import mlflow.xgboost

from preprocess import run_pipeline
from models import (
    RandomForestIDS, SVMIDS, LSTMIDS, AutoencoderIDS, XGBoostIDS,
    TIMESTEPS,
)
from evaluate import compute_all_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MLFLOW_TRACKING_URI = os.path.join(ROOT_DIR, "mlruns")
EXPERIMENT_NAME = "ids-zero-trust"

mlflow.set_tracking_uri(f"file:///{MLFLOW_TRACKING_URI}")
mlflow.set_experiment(EXPERIMENT_NAME)


def load_data():
    """Load preprocessed data, running the pipeline if needed."""
    data_dir = os.path.join(ROOT_DIR, "data")
    X_train_path = os.path.join(data_dir, "X_train.npy")
    if not os.path.exists(X_train_path):
        logger.info("Preprocessed data not found. Running full pipeline...")
        run_pipeline()

    X_train = np.load(os.path.join(data_dir, "X_train.npy"))
    X_test = np.load(os.path.join(data_dir, "X_test.npy"))
    y_train = np.load(os.path.join(data_dir, "y_train.npy"))
    y_test = np.load(os.path.join(data_dir, "y_test.npy"))
    logger.info(
        "Loaded data — train: %s, test: %s", X_train.shape, X_test.shape
    )
    return X_train, X_test, y_train, y_test


def train_random_forest(X_train, y_train, X_test, y_test):
    """Train RandomForestIDS with MLflow tracking."""
    with mlflow.start_run(run_name="RandomForest", tags={"model": "rf"}):
        mlflow.log_params({
            "n_estimators": 200, "criterion": "gini",
            "max_depth": None, "algorithm": "RandomForest",
        })
        model = RandomForestIDS(random_state=42)
        model.train(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        metrics = compute_all_metrics(y_test, y_pred, y_proba)

        mlflow.log_metrics({
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1"],
            "fpr": metrics["fpr"],
            "training_time_s": model.training_time,
        })
        mlflow.log_param("training_time_s", model.training_time)

        # Cross-validation
        cv = model.cross_validate_10fold(X_train, y_train)
        mlflow.log_metrics({
            "cv_accuracy_mean": cv["accuracy_mean"],
            "cv_accuracy_std": cv["accuracy_std"],
            "cv_f1_mean": cv["f1_mean"],
        })

        # Save and log model artifact
        model_path = model.save(name="random_forest")
        mlflow.log_artifact(model_path, artifact_path="models")
        mlflow.sklearn.log_model(
            model.model, "model",
            registered_model_name="RandomForestIDS",
        )
        logger.info("RandomForest — accuracy: %.4f, F1: %.4f",
                     metrics["accuracy"], metrics["f1"])
        return metrics


def train_svm(X_train, y_train, X_test, y_test):
    """Train SVMIDS with MLflow tracking."""
    with mlflow.start_run(run_name="SVM", tags={"model": "svm"}):
        mlflow.log_params({
            "kernel": "rbf", "C": 10, "gamma": 0.001,
            "probability": True, "algorithm": "SVM",
        })
        model = SVMIDS()
        model.train(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        metrics = compute_all_metrics(y_test, y_pred, y_proba)

        mlflow.log_metrics({
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1"],
            "fpr": metrics["fpr"],
            "training_time_s": model.training_time,
        })

        model_path = model.save(name="svm")
        mlflow.log_artifact(model_path, artifact_path="models")
        mlflow.sklearn.log_model(
            model.model, "model",
            registered_model_name="SVMIDS",
        )
        logger.info("SVM — accuracy: %.4f, F1: %.4f",
                     metrics["accuracy"], metrics["f1"])
        return metrics


def train_lstm(X_train, y_train, X_test, y_test):
    """Train LSTMIDS with MLflow tracking."""
    with mlflow.start_run(run_name="LSTM", tags={"model": "lstm"}):
        mlflow.log_params({
            "layers": 2, "units": 128, "dropout": 0.3,
            "optimizer": "adam", "epochs": 50, "patience": 5,
            "batch_size": 256, "timesteps": TIMESTEPS,
            "algorithm": "LSTM",
        })
        model = LSTMIDS(n_classes=2, timesteps=TIMESTEPS)
        model.train(X_train, y_train, validation_split=0.1,
                     batch_size=256, epochs=50)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        # Align lengths for metric computation
        n_seq = (len(y_test) // TIMESTEPS) * TIMESTEPS
        metrics = compute_all_metrics(y_test[:n_seq:TIMESTEPS],
                                       y_pred, y_proba)

        mlflow.log_metrics({
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1"],
            "fpr": metrics["fpr"],
            "training_time_s": model.training_time,
        })

        # Log training history
        if model.history is not None:
            for epoch, (loss, val_loss) in enumerate(zip(
                model.history.history.get("loss", []),
                model.history.history.get("val_loss", []),
            )):
                mlflow.log_metrics({
                    "train_loss": loss, "val_loss": val_loss,
                }, step=epoch)

        model_path = model.save(name="lstm")
        mlflow.log_artifact(model_path, artifact_path="models")
        mlflow.keras.log_model(
            model.model, "model",
            registered_model_name="LSTMIDS",
        )
        logger.info("LSTM — accuracy: %.4f, F1: %.4f",
                     metrics["accuracy"], metrics["f1"])
        return metrics


def train_autoencoder(X_train, y_train, X_test, y_test):
    """Train AutoencoderIDS with MLflow tracking."""
    with mlflow.start_run(run_name="Autoencoder", tags={"model": "autoencoder"}):
        mlflow.log_params({
            "architecture": "41-32-16-8", "threshold_percentile": 95,
            "epochs": 50, "batch_size": 256,
            "algorithm": "Autoencoder",
        })
        model = AutoencoderIDS(input_dim=X_train.shape[1])
        model.train(X_train, y_train, epochs=50, batch_size=256)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        metrics = compute_all_metrics(y_test, y_pred, y_proba)

        mlflow.log_metrics({
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1"],
            "fpr": metrics["fpr"],
            "training_time_s": model.training_time,
            "reconstruction_threshold": model.threshold,
        })

        model_path = model.save(name="autoencoder")
        mlflow.log_artifact(model_path, artifact_path="models")
        mlflow.keras.log_model(
            model.model, "model",
            registered_model_name="AutoencoderIDS",
        )
        logger.info("Autoencoder — accuracy: %.4f, F1: %.4f",
                     metrics["accuracy"], metrics["f1"])
        return metrics


def train_xgboost(X_train, y_train, X_test, y_test):
    """Train XGBoostIDS with MLflow tracking."""
    with mlflow.start_run(run_name="XGBoost", tags={"model": "xgboost"}):
        mlflow.log_params({
            "n_estimators": 500, "learning_rate": 0.05,
            "max_depth": 6, "subsample": 0.8,
            "colsample_bytree": 0.8, "algorithm": "XGBoost",
        })
        model = XGBoostIDS(random_state=42)
        model.train(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        metrics = compute_all_metrics(y_test, y_pred, y_proba)

        mlflow.log_metrics({
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1"],
            "fpr": metrics["fpr"],
            "training_time_s": model.training_time,
        })

        # Cross-validation
        cv = model.cross_validate_10fold(X_train, y_train)
        mlflow.log_metrics({
            "cv_accuracy_mean": cv["accuracy_mean"],
            "cv_accuracy_std": cv["accuracy_std"],
            "cv_f1_mean": cv["f1_mean"],
        })

        model_path = model.save(name="xgboost")
        mlflow.log_artifact(model_path, artifact_path="models")
        mlflow.xgboost.log_model(
            model.model, "model",
            registered_model_name="XGBoostIDS",
        )
        logger.info("XGBoost — accuracy: %.4f, F1: %.4f",
                     metrics["accuracy"], metrics["f1"])
        return metrics


TRAINERS = {
    "random_forest": train_random_forest,
    "svm": train_svm,
    "lstm": train_lstm,
    "autoencoder": train_autoencoder,
    "xgboost": train_xgboost,
}


def main():
    parser = argparse.ArgumentParser(
        description="MLflow-tracked IDS model training"
    )
    parser.add_argument(
        "--model", type=str, default=None,
        choices=list(TRAINERS.keys()),
        help="Train a single model (default: all 5)",
    )
    args = parser.parse_args()

    logger.info("MLflow tracking URI: %s", MLFLOW_TRACKING_URI)
    logger.info("Experiment: %s", EXPERIMENT_NAME)

    X_train, X_test, y_train, y_test = load_data()

    results = {}
    if args.model:
        results[args.model] = TRAINERS[args.model](
            X_train, y_train, X_test, y_test
        )
    else:
        for name, trainer in TRAINERS.items():
            logger.info("--- Training %s ---", name)
            try:
                results[name] = trainer(X_train, y_train, X_test, y_test)
            except Exception as exc:
                logger.error("Failed to train %s: %s", name, exc)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Training complete. Results:")
    for name, metrics in results.items():
        logger.info(
            "  %-16s  Acc: %.4f  F1: %.4f  FPR: %.4f  Time: %.1fs",
            name, metrics["accuracy"], metrics["f1"],
            metrics["fpr"], metrics.get("training_time_s", 0),
        )
    logger.info("=" * 60)
    logger.info("Launch MLflow UI: mlflow ui --port 5000")


if __name__ == "__main__":
    main()
