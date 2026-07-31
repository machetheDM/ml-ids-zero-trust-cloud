"""
serve.py
========
FastAPI Inference Endpoint for ML-IDS Zero Trust Cloud.

Serves the best-performing model (LSTM, 98.1% accuracy) as a REST API.
Accepts raw network flow features and returns intrusion classification
with confidence scores.

Endpoints:
  POST /predict        — classify a batch of network flows
  POST /predict/single — classify a single network flow
  GET  /health         — model health + metadata
  GET  /models         — list available models

Usage:
  uvicorn api.serve:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import sys
import time
import logging
import numpy as np
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from src.models import (
    RandomForestIDS, SVMIDS, LSTMIDS, AutoencoderIDS, XGBoostIDS,
    TIMESTEPS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

MODELS_DIR = os.path.join(ROOT_DIR, "results", "models")

app = FastAPI(
    title="ML-IDS Zero Trust Cloud — Inference API",
    description="Intrusion detection inference for cloud network security.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Model registry (lazy-loading)
# ---------------------------------------------------------------------------

MODEL_REGISTRY: dict[str, object] = {}
DEFAULT_MODEL = "lstm"
N_FEATURES = 25

MODEL_META = {
    "random_forest": {"type": "RandomForest", "accuracy": 0.968, "fpr": 0.021},
    "svm": {"type": "SVM", "accuracy": 0.942, "fpr": 0.034},
    "lstm": {"type": "LSTM", "accuracy": 0.981, "fpr": 0.018},
    "autoencoder": {"type": "Autoencoder", "accuracy": 0.915, "fpr": 0.042},
    "xgboost": {"type": "XGBoost", "accuracy": 0.973, "fpr": 0.019},
}


def load_model(name: str):
    """Lazy-load a model from disk."""
    if name in MODEL_REGISTRY:
        return MODEL_REGISTRY[name]

    loaders = {
        "random_forest": lambda: RandomForestIDS.load(name="random_forest"),
        "svm": lambda: SVMIDS.load(name="svm"),
        "lstm": lambda: LSTMIDS.load(name="lstm", n_classes=2),
        "autoencoder": lambda: AutoencoderIDS.load(name="autoencoder"),
        "xgboost": lambda: XGBoostIDS.load(name="xgboost"),
    }

    if name not in loaders:
        available = ", ".join(loaders.keys())
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model '{name}'. Available: {available}",
        )

    try:
        model = loaders[name]()
        MODEL_REGISTRY[name] = model
        logger.info("Loaded model: %s", name)
        return model
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail=f"Model '{name}' not found on disk. "
                   f"Run training first: python src/train_mlflow.py",
        )


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    features: list[list[float]] = Field(
        ...,
        description="List of feature vectors. Each vector must have 25 features "
                    "(post-RFECV). Shape: [n_samples, 25]",
        min_length=1,
    )
    model: str = Field(
        default=DEFAULT_MODEL,
        description="Model to use: random_forest, svm, lstm, autoencoder, xgboost",
    )


class SinglePredictRequest(BaseModel):
    features: list[float] = Field(
        ...,
        description="Single feature vector with 25 features (post-RFECV).",
        min_length=N_FEATURES,
        max_length=N_FEATURES,
    )
    model: str = Field(default=DEFAULT_MODEL)


class PredictionResult(BaseModel):
    prediction: int = Field(..., description="0 = normal, 1 = attack")
    confidence: float = Field(..., description="Probability of predicted class")
    probabilities: dict[str, float] = Field(
        ..., description="normal_probability, attack_probability"
    )
    model_used: str
    inference_time_ms: float


class BatchPredictionResult(BaseModel):
    predictions: list[int]
    confidences: list[float]
    probabilities: list[dict[str, float]]
    model_used: str
    n_samples: int
    inference_time_ms: float
    attack_rate: float


class HealthResponse(BaseModel):
    status: str
    models_available: list[str]
    default_model: str
    n_features_expected: int
    uptime_seconds: float


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

START_TIME = time.time()


@app.get("/health", response_model=HealthResponse)
def health():
    """Health check with model availability."""
    available = []
    for name in MODEL_META:
        path = os.path.join(MODELS_DIR, f"{name}.pkl")
        h5_path = os.path.join(MODELS_DIR, f"{name}.h5")
        if os.path.exists(path) or os.path.exists(h5_path):
            available.append(name)

    return HealthResponse(
        status="healthy" if available else "degraded — no models loaded",
        models_available=available,
        default_model=DEFAULT_MODEL,
        n_features_expected=N_FEATURES,
        uptime_seconds=round(time.time() - START_TIME, 2),
    )


@app.get("/models")
def list_models():
    """List all models with metadata."""
    available = []
    for name, meta in MODEL_META.items():
        path = os.path.join(MODELS_DIR, f"{name}.pkl")
        h5_path = os.path.join(MODELS_DIR, f"{name}.h5")
        available.append({
            "name": name,
            "type": meta["type"],
            "accuracy": meta["accuracy"],
            "fpr": meta["fpr"],
            "loaded": os.path.exists(path) or os.path.exists(h5_path),
        })
    return {"models": available, "default": DEFAULT_MODEL}


@app.post("/predict", response_model=BatchPredictionResult)
def predict_batch(req: PredictRequest):
    """Classify a batch of network flows."""
    t0 = time.time()

    # Validate feature dimensions
    X = np.array(req.features, dtype=np.float32)
    if X.shape[1] != N_FEATURES:
        raise HTTPException(
            status_code=400,
            detail=f"Expected {N_FEATURES} features per sample, got {X.shape[1]}",
        )

    model = load_model(req.model)

    # Predict
    predictions = model.predict(X).tolist()
    proba = model.predict_proba(X)

    # Build response
    confidences = []
    probabilities = []
    for i in range(len(predictions)):
        p_normal = float(proba[i][0])
        p_attack = float(proba[i][1])
        confidences.append(max(p_normal, p_attack))
        probabilities.append({
            "normal": round(p_normal, 6),
            "attack": round(p_attack, 6),
        })

    elapsed_ms = (time.time() - t0) * 1000
    attack_rate = sum(predictions) / len(predictions)

    logger.info(
        "Batch predict: %d samples, %.1fms, attack_rate=%.2f%%",
        len(predictions), elapsed_ms, attack_rate * 100,
    )

    return BatchPredictionResult(
        predictions=predictions,
        confidences=confidences,
        probabilities=probabilities,
        model_used=req.model,
        n_samples=len(predictions),
        inference_time_ms=round(elapsed_ms, 2),
        attack_rate=round(attack_rate, 4),
    )


@app.post("/predict/single", response_model=PredictionResult)
def predict_single(req: SinglePredictRequest):
    """Classify a single network flow."""
    t0 = time.time()

    X = np.array([req.features], dtype=np.float32)
    model = load_model(req.model)

    pred = int(model.predict(X)[0])
    proba = model.predict_proba(X)[0]
    p_normal = float(proba[0])
    p_attack = float(proba[1])

    elapsed_ms = (time.time() - t0) * 1000

    return PredictionResult(
        prediction=pred,
        confidence=max(p_normal, p_attack),
        probabilities={
            "normal": round(p_normal, 6),
            "attack": round(p_attack, 6),
        },
        model_used=req.model,
        inference_time_ms=round(elapsed_ms, 2),
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.serve:app", host="0.0.0.0", port=8000, reload=True)
