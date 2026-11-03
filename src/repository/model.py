"""Persistence for the fitted churn pipeline and its training metadata."""

import json
import os
import pickle
import tempfile
import uuid
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

from repository.dataset import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS, TARGET_COLUMN

MODEL_DIRECTORY = Path(__file__).resolve().parents[2] / "models"
MODEL_MANIFEST_PATH = MODEL_DIRECTORY / "churn_model.json"


MODEL_PARAMETERS = {
    "logreg": {"C", "max_iter", "solver", "class_weight"},
    "random_forest": {"n_estimators", "max_depth", "min_samples_split", "min_samples_leaf", "class_weight"},
}


def train_churn_model(
    training_data: pd.DataFrame,
    model_type: str = "logreg",
    hyperparameters: dict[str, Any] | None = None,
) -> Pipeline:
    """Fit and return a preprocessing/classification pipeline on training data."""
    if training_data is None or training_data.empty:
        raise ValueError("Training dataset is empty")
    required = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS + [TARGET_COLUMN]
    missing = [column for column in required if column not in training_data.columns]
    if missing:
        raise ValueError(f"Training dataset is missing columns: {', '.join(missing)}")
    if training_data[required].isna().any().any():
        raise ValueError("Training dataset contains missing values")

    X = training_data[NUMERIC_COLUMNS + CATEGORICAL_COLUMNS]
    y = training_data[TARGET_COLUMN]
    if y.nunique() < 2:
        raise ValueError("Training data must contain both churn classes")

    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_COLUMNS),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLUMNS),
        ]
    )
    if model_type not in MODEL_PARAMETERS:
        raise ValueError("model_type must be 'logreg' or 'random_forest'")
    params = dict(hyperparameters or {})
    unsupported = sorted(set(params) - MODEL_PARAMETERS[model_type])
    if unsupported:
        raise ValueError(f"Unsupported {model_type} hyperparameters: {', '.join(unsupported)}")
    defaults = (
        {"C": 1.0, "max_iter": 1000, "solver": "lbfgs", "class_weight": None}
        if model_type == "logreg"
        else {"n_estimators": 100, "random_state": 42}
    )
    defaults.update(params)
    estimator = LogisticRegression(**defaults) if model_type == "logreg" else RandomForestClassifier(**defaults)
    pipeline = Pipeline(steps=[("preprocessor", preprocessing), ("classifier", estimator)])
    pipeline.fit(X, y)
    pipeline.model_type = model_type
    pipeline.model_hyperparameters = {key: getattr(estimator, key) for key in MODEL_PARAMETERS[model_type]}
    return pipeline

def predict_churn(pipeline: Pipeline, features: pd.DataFrame) -> list[dict[str, Any]]:
    """Run inference using the persisted training pipeline and its class ordering."""
    feature_columns = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS
    missing = [column for column in feature_columns if column not in features.columns]
    if missing:
        raise ValueError(f"Prediction data is missing columns: {', '.join(missing)}")
    ordered = features[feature_columns]
    predicted = pipeline.predict(ordered)
    probabilities = pipeline.predict_proba(ordered)
    classes = pipeline.named_steps["classifier"].classes_
    return [
        {
            "predicted_class": int(label),
            "class_probabilities": {
                str(int(class_label)): float(probability)
                for class_label, probability in zip(classes, row_probabilities)
            },
        }
        for label, row_probabilities in zip(predicted, probabilities)
    ]

def _write_temp(directory: Path, suffix: str, writer) -> Path:
    """Write and fsync a temporary file in the target directory."""
    fd, name = tempfile.mkstemp(prefix=".churn-", suffix=suffix, dir=directory)
    path = Path(name)
    try:
        with os.fdopen(fd, "wb") as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise
    return path


def save_churn_model(
    pipeline: Pipeline,
    trained_at: str,
    metrics: dict[str, float],
    model_type: str | None = None,
    hyperparameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist a complete model revision, then atomically publish its manifest."""
    if not isinstance(pipeline, Pipeline):
        raise ValueError("Only a fitted scikit-learn Pipeline can be saved")
    try:
        check_is_fitted(pipeline)
    except Exception as exc:
        raise ValueError("Cannot save an unfitted churn model") from exc
    MODEL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    revision = uuid.uuid4().hex
    model_name = f"churn_model_{revision}.pkl"
    metadata_name = f"churn_metadata_{revision}.json"
    model_path = MODEL_DIRECTORY / model_name
    metadata_path = MODEL_DIRECTORY / metadata_name
    manifest_temp: Path | None = None
    model_temp: Path | None = None
    metadata_temp: Path | None = None
    metadata: dict[str, Any] = {
        "available": True,
        "trained_at": trained_at,
        "metrics": {key: float(value) for key, value in metrics.items()},
        "model_type": model_type or getattr(pipeline, "model_type", "logreg"),
        "hyperparameters": hyperparameters if hyperparameters is not None else getattr(pipeline, "model_hyperparameters", {}),
    }
    try:
        model_temp = _write_temp(
            MODEL_DIRECTORY,
            ".pkl.tmp",
            lambda stream: pickle.dump(pipeline, stream, protocol=pickle.HIGHEST_PROTOCOL),
        )
        metadata_temp = _write_temp(
            MODEL_DIRECTORY,
            ".json.tmp",
            lambda stream: stream.write(json.dumps(metadata, indent=2).encode("utf-8")),
        )
        os.replace(model_temp, model_path)
        model_temp = None
        os.replace(metadata_temp, metadata_path)
        metadata_temp = None
        manifest = {"model_file": model_name, "metadata_file": metadata_name}
        manifest_temp = _write_temp(
            MODEL_DIRECTORY,
            ".manifest.tmp",
            lambda stream: stream.write(json.dumps(manifest).encode("utf-8")),
        )
        os.replace(manifest_temp, MODEL_MANIFEST_PATH)
        manifest_temp = None
        return metadata
    finally:
        for temp_path in (model_temp, metadata_temp, manifest_temp):
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)


def load_churn_model() -> tuple[Pipeline, dict[str, Any]] | None:
    """Load the published pipeline and status; return None when never trained."""
    if not MODEL_MANIFEST_PATH.exists():
        return None
    try:
        manifest = json.loads(MODEL_MANIFEST_PATH.read_text(encoding="utf-8"))
        model_name = manifest["model_file"]
        metadata_name = manifest["metadata_file"]
        if Path(model_name).name != model_name or Path(metadata_name).name != metadata_name:
            raise ValueError("Model manifest contains an invalid artifact path")
        with (MODEL_DIRECTORY / model_name).open("rb") as stream:
            pipeline = pickle.load(stream)
        metadata = json.loads((MODEL_DIRECTORY / metadata_name).read_text(encoding="utf-8"))
        if not isinstance(pipeline, Pipeline) or not isinstance(metadata, dict):
            raise ValueError("Saved churn model or metadata has an invalid format")
        if metadata.get("available") is not True:
            raise ValueError("Saved churn model or metadata has an invalid format")
        if not isinstance(metadata.get("metrics"), dict) or not metadata.get("trained_at"):
            raise ValueError("Saved churn model metadata is incomplete")
        check_is_fitted(pipeline)
        # Keep pre-Task-8 model artifacts readable and expose their configuration.
        metadata.setdefault("model_type", getattr(pipeline, "model_type", "logreg"))
        metadata.setdefault("hyperparameters", getattr(pipeline, "model_hyperparameters", {}))
        return pipeline, metadata
    except (OSError, pickle.PickleError, EOFError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not load saved churn model: {exc}") from exc
