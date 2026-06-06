"""
models.py
=========
ML model definitions for ML-IDS Zero Trust Cloud Research

All hyperparameters sourced directly from the research paper (§5.3):
  "Machine Learning-Based Intrusion Detection for Cloud Network Security:
   A Zero Trust Architecture Approach"
Author: Dingaan Mahlatse Machethe — EC-Council University, ECCU500

Five algorithms evaluated:
  1. Random Forest  — 200 trees, Gini, max_depth=None
  2. SVM            — RBF kernel, C=10, gamma=0.001
  3. LSTM           — 2-layer, 128 units, dropout=0.3, Adam, 50 epochs, patience=5
  4. Autoencoder    — 41→32→16→8→16→32→41, threshold=95th percentile
  5. XGBoost        — 500 estimators, lr=0.05, max_depth=6, subsample=0.8
"""

import numpy as np
import joblib
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb


# ---------------------------------------------------------------------------
# Placeholder imports for TensorFlow/Keras — imported lazily to avoid
# hard dependency when only sklearn models are needed
# ---------------------------------------------------------------------------

def _get_keras():
    try:
        import tensorflow as tf
        from tensorflow import keras
        return keras
    except ImportError as e:
        raise ImportError(
            "TensorFlow is required for LSTM and Autoencoder models. "
            "Install with: pip install tensorflow"
        ) from e


# ---------------------------------------------------------------------------
# 1. Random Forest
# Paper §5.3: 200 trees, Gini impurity, max depth unrestricted
# ---------------------------------------------------------------------------

def build_random_forest(random_state: int = 42) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=200,
        criterion="gini",
        max_depth=None,
        n_jobs=-1,
        random_state=random_state,
    )


# ---------------------------------------------------------------------------
# 2. Support Vector Machine
# Paper §5.3: RBF kernel, C=10, gamma=0.001
# ---------------------------------------------------------------------------

def build_svm() -> SVC:
    return SVC(
        kernel="rbf",
        C=10,
        gamma=0.001,
        probability=True,
        random_state=42,
    )


# ---------------------------------------------------------------------------
# 3. LSTM
# Paper §5.3: Two-layer, 128 units/layer, dropout=0.3, Adam, 50 epochs,
#             early stopping (patience=5), sequences of 20 network flows
# ---------------------------------------------------------------------------

def build_lstm(input_shape: tuple, n_classes: int = 2):
    """
    Build a two-layer LSTM model.

    Args:
        input_shape: (timesteps, features) — paper uses sequences of 20 flows
        n_classes: number of output classes (2 for binary classification)
    """
    keras = _get_keras()

    model = keras.Sequential([
        keras.layers.LSTM(
            128,
            return_sequences=True,
            input_shape=input_shape,
            name="lstm_layer_1",
        ),
        keras.layers.Dropout(0.3, name="dropout_1"),
        keras.layers.LSTM(
            128,
            return_sequences=False,
            name="lstm_layer_2",
        ),
        keras.layers.Dropout(0.3, name="dropout_2"),
        keras.layers.Dense(64, activation="relu", name="dense_1"),
        keras.layers.Dense(
            1 if n_classes == 2 else n_classes,
            activation="sigmoid" if n_classes == 2 else "softmax",
            name="output",
        ),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss="binary_crossentropy" if n_classes == 2 else "sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def get_lstm_callbacks(patience: int = 5):
    """Early stopping callback — paper: patience=5."""
    keras = _get_keras()
    return [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            verbose=1,
        ),
    ]


def reshape_for_lstm(X: np.ndarray, timesteps: int = 20) -> np.ndarray:
    """
    Reshape flat feature vectors into LSTM sequences.

    Paper §5.3: "Input sequences of 20 consecutive network flows are
    constructed to enable temporal pattern recognition."

    Pads the array so total samples is divisible by timesteps.
    """
    n_samples, n_features = X.shape
    n_sequences = n_samples // timesteps
    X_trimmed = X[: n_sequences * timesteps]
    return X_trimmed.reshape(n_sequences, timesteps, n_features)


# ---------------------------------------------------------------------------
# 4. Autoencoder
# Paper §5.3: Encoder 41→32→16→8, symmetric decoder,
#             threshold = 95th percentile of normal traffic reconstruction error
# ---------------------------------------------------------------------------

def build_autoencoder(input_dim: int = 41):
    """
    Build a symmetric autoencoder for anomaly detection.

    Architecture: input_dim → 32 → 16 → 8 (bottleneck) → 16 → 32 → input_dim
    Trained exclusively on normal traffic samples.
    """
    keras = _get_keras()

    # Encoder
    inputs = keras.Input(shape=(input_dim,), name="encoder_input")
    x = keras.layers.Dense(32, activation="relu", name="enc_32")(inputs)
    x = keras.layers.Dense(16, activation="relu", name="enc_16")(x)
    encoded = keras.layers.Dense(8, activation="relu", name="bottleneck")(x)

    # Decoder (symmetric)
    x = keras.layers.Dense(16, activation="relu", name="dec_16")(encoded)
    x = keras.layers.Dense(32, activation="relu", name="dec_32")(x)
    decoded = keras.layers.Dense(input_dim, activation="sigmoid", name="decoder_output")(x)

    autoencoder = keras.Model(inputs, decoded, name="autoencoder")
    autoencoder.compile(optimizer="adam", loss="mse")
    return autoencoder


def compute_reconstruction_threshold(
    autoencoder,
    X_normal: np.ndarray,
    percentile: float = 95.0,
) -> float:
    """
    Compute the reconstruction error threshold.

    Paper §5.3: "Reconstruction error threshold set at 95th percentile of
    normal traffic reconstruction error distribution."
    """
    reconstructions = autoencoder.predict(X_normal, verbose=0)
    errors = np.mean(np.square(X_normal - reconstructions), axis=1)
    threshold = float(np.percentile(errors, percentile))
    return threshold


def predict_autoencoder(
    autoencoder,
    X: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Classify samples as 0 (normal) or 1 (attack) based on reconstruction error."""
    reconstructions = autoencoder.predict(X, verbose=0)
    errors = np.mean(np.square(X - reconstructions), axis=1)
    return (errors > threshold).astype(int)


# ---------------------------------------------------------------------------
# 5. XGBoost
# Paper §5.3: 500 estimators, lr=0.05, max_depth=6, subsample=0.8
# ---------------------------------------------------------------------------

def build_xgboost(random_state: int = 42) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1,
    )


# ---------------------------------------------------------------------------
# Model persistence helpers
# ---------------------------------------------------------------------------

MODELS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "models",
)


def save_sklearn_model(model, name: str, output_dir: str = MODELS_DIR) -> str:
    """Save a scikit-learn / XGBoost model as a .pkl file."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{name}.pkl")
    joblib.dump(model, path)
    return path


def load_sklearn_model(name: str, model_dir: str = MODELS_DIR):
    """Load a scikit-learn / XGBoost model from a .pkl file."""
    path = os.path.join(model_dir, f"{name}.pkl")
    return joblib.load(path)


def save_keras_model(model, name: str, output_dir: str = MODELS_DIR) -> str:
    """Save a Keras model as an .h5 file."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{name}.h5")
    model.save(path)
    return path
