from datetime import datetime, timezone
import logging

from fastapi import HTTPException
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import pandas as pd
from repository import dataset as dataset_repository
from repository import model as model_repository
from repository import training_history as history_repository

_loaded_model = None
_training_status = {
    "available": False, "trained_at": None, "metrics": None,
    "model_type": None, "hyperparameters": {},
}
_load_error: str | None = None
logger = logging.getLogger(__name__)


def is_model_available() -> bool:
    return _loaded_model is not None and _training_status.get("available", False) and _load_error is None


def get_feature_schema() -> dict:
    """Describe the ordered prediction input schema for API clients."""
    descriptions = {
        "monthly_fee": {"type": "number"},
        "usage_hours": {"type": "number"},
        "support_requests": {"type": "integer"},
        "account_age_months": {"type": "integer"},
        "failed_payments": {"type": "integer"},
        "autopay_enabled": {"type": "integer", "binary": [0, 1]},
        "region": {"type": "string", "allowed_values": ["europe", "asia", "america", "africa"]},
        "device_type": {"type": "string", "allowed_values": ["mobile", "desktop", "tablet"]},
        "payment_method": {"type": "string", "allowed_values": ["card", "paypal", "crypto"]},
    }
    from repository.dataset import NUMERIC_COLUMNS, CATEGORICAL_COLUMNS

    names = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS
    return {
        "features": [
            {"name": name, **descriptions[name]} for name in names
        ],
        "target": "churn (not accepted as a prediction input)",
    }


def load_saved_model() -> None:
    """Populate runtime state from disk; an absent model is a normal first start."""
    global _loaded_model, _training_status, _load_error
    try:
        saved = model_repository.load_churn_model()
        if saved is None:
            _loaded_model = None
            _training_status = {
                "available": False, "trained_at": None, "metrics": None,
                "model_type": None, "hyperparameters": {},
            }
            _load_error = None
        else:
            _loaded_model, _training_status = saved
            _load_error = None
            logger.info("Saved churn model loaded: model_type=%s", _training_status.get("model_type"))
    except (OSError, ValueError) as exc:
        _loaded_model = None
        _training_status = {
            "available": False, "trained_at": None, "metrics": None,
            "model_type": None, "hyperparameters": {},
        }
        _load_error = str(exc)
        logger.exception("Failed to load saved churn model")


def get_model_status() -> dict:
    if _load_error:
        raise HTTPException(status_code=503, detail={
            "code": "MODEL_UNAVAILABLE", "message": "The saved churn model is unavailable.",
            "details": [{"field": "model", "issue": "Retrain the model or restore a valid saved model."}],
        })
    if _loaded_model is None or not _training_status.get("available", False):
        raise HTTPException(status_code=503, detail={
            "code": "MODEL_NOT_TRAINED", "message": "No trained churn model is available.",
            "details": [{"field": "model", "issue": "Call POST /model/train before requesting model status."}],
        })
    return dict(_training_status)

def predict_customers(records: list[dict]) -> list[dict]:
    """Predict churn for validated feature records using the available model."""
    if _loaded_model is None:
        if _load_error:
            raise HTTPException(status_code=503, detail={
                "code": "MODEL_UNAVAILABLE", "message": "The saved churn model is unavailable.",
                "details": [{"field": "model", "issue": "Train a model or restore a valid saved model."}],
            })
        raise HTTPException(status_code=503, detail={
            "code": "MODEL_NOT_TRAINED", "message": "No trained churn model is available.",
            "details": [{"field": "model", "issue": "Call POST /model/train before predicting."}],
        })
    try:
        return model_repository.predict_churn(_loaded_model, pd.DataFrame(records))
    except Exception as exc:
        logger.exception("Churn prediction failed: records=%d", len(records))
        raise HTTPException(status_code=500, detail={
            "code": "PREDICTION_FAILED", "message": "The service could not generate churn predictions.",
            "details": [],
        }) from exc

def train_model_and_evaluate(config) -> dict:
    """Train on a reproducible training split and report held-out metrics."""
    logger.info("Churn model training started: model_type=%s", config.model_type)
    try:
        X_train, X_test, y_train, y_test = dataset_repository.prepare_train_test_split()
    except (FileNotFoundError, ValueError) as exc:
        if isinstance(exc, ValueError) and "contains no rows" in str(exc):
            raise HTTPException(status_code=422, detail={
                "code": "EMPTY_DATASET", "message": "The training dataset is empty.",
                "details": [{"field": "dataset", "issue": "Provide at least one valid churn row before training."}],
            }) from exc
        raise HTTPException(status_code=503, detail={
            "code": "TRAINING_DATA_UNAVAILABLE", "message": "Training data is unavailable or invalid.",
            "details": [{"field": "dataset", "issue": "Check that the configured dataset exists and matches the churn schema."}],
        }) from exc
    training_data = X_train.copy()
    training_data[dataset_repository.TARGET_COLUMN] = y_train
    try:
        pipeline = model_repository.train_churn_model(
            training_data, config.model_type, config.hyperparameters
        )
        predictions = pipeline.predict(X_test)
        metrics = {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "f1": float(f1_score(y_test, predictions, average="binary", pos_label=1, zero_division=0)),
        }
        try:
            probabilities = pipeline.predict_proba(X_test)
            classes = list(pipeline.named_steps["classifier"].classes_)
            positive_index = classes.index(1)
            metrics["roc_auc"] = float(roc_auc_score(y_test, probabilities[:, positive_index]))
        except Exception:
            # ROC AUC is undefined for a one-class held-out set or unavailable scores.
            pass
        trained_at = datetime.now(timezone.utc).isoformat()
        # Persist history before publishing the model manifest. If history cannot
        # be saved, the currently published model remains untouched.
        previous_history = history_repository.load_training_history()
        history_repository.record_training_run({
            "trained_at": trained_at,
            "model_type": config.model_type,
            "hyperparameters": pipeline.model_hyperparameters,
            "metrics": metrics,
        })
        try:
            persisted_status = model_repository.save_churn_model(
                pipeline, trained_at, metrics, config.model_type, pipeline.model_hyperparameters
            )
        except Exception:
            # Do not leave a successful-looking history entry if model
            # persistence failed before the model was published.
            try:
                history_repository.write_training_history(previous_history)
            except Exception:
                logger.exception("Could not roll back training history after model persistence failed")
            raise
    except Exception as exc:
        logger.exception("Churn model training failed: model_type=%s", config.model_type)
        raise HTTPException(status_code=500, detail={
            "code": "TRAINING_FAILED", "message": "The service could not train or save the churn model.",
            "details": [],
        }) from exc

    # Publish the new pipeline in memory only after persistence commits.
    global _loaded_model, _training_status, _load_error
    _loaded_model = pipeline
    _training_status = persisted_status
    _load_error = None
    logger.info("Churn model training completed: model_type=%s", config.model_type)
    return persisted_status


def get_training_metrics(limit: int = 20, model_type: str | None = None) -> dict:
    """Return the latest successful training metrics and recent matching runs."""
    try:
        records = history_repository.load_training_history()
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=503, detail={
            "code": "TRAINING_HISTORY_UNAVAILABLE",
            "message": "Training history is unavailable.",
            "details": [{"field": "history", "issue": "Check the configured training history file."}],
        }) from exc
    if model_type:
        records = [record for record in records if record.get("model_type") == model_type]
    if not records:
        raise HTTPException(status_code=404, detail={
            "code": "TRAINING_HISTORY_EMPTY",
            "message": "No successful training record is available.",
            "details": [{"field": "history", "issue": "Train a model before requesting metrics."}],
        })
    recent = list(reversed(records[-limit:]))
    latest = recent[0]
    return {
        "trained_at": latest["trained_at"],
        "model_type": latest["model_type"],
        "hyperparameters": latest["hyperparameters"],
        "metrics": latest["metrics"],
        "history": recent,
    }
    
