from fastapi import APIRouter, Body, Query

from service import model as model_service
from schemas import TrainingConfigChurn

router = APIRouter()


@router.post("/model/train")
def train_model(
    config: TrainingConfigChurn = Body(
        openapi_examples={
            "logreg_defaults": {
                "summary": "Train with default logistic regression settings",
                "value": {
                    "model_type": "logreg",
                    "hyperparameters": {
                        "C": 1.0,
                        "max_iter": 1000,
                        "solver": "lbfgs",
                        "class_weight": None,
                    },
                },
            },
            "random_forest_defaults": {
                "summary": "Train with default random forest settings",
                "value": {
                    "model_type": "random_forest",
                    "hyperparameters": {"n_estimators": 100},
                },
            },
        }
    ),
    responses={
        422: {
            "description": "Training dataset is empty or invalid.",
            "content": {"application/json": {"example": {
                "code": "EMPTY_DATASET", "message": "The training dataset is empty.",
                "details": [{"field": "dataset", "issue": "Provide at least one valid churn row before training."}],
            }}},
        },
        503: {
            "description": "Training data is unavailable.",
            "content": {"application/json": {"example": {
                "code": "TRAINING_DATA_UNAVAILABLE", "message": "Training data is unavailable or invalid.",
                "details": [{"field": "dataset", "issue": "Check that the configured dataset exists and matches the churn schema."}],
            }}},
        },
    },
):
    return model_service.train_model_and_evaluate(config)


@router.get(
    "/model/status",
    responses={
        503: {
            "description": "No trained model is available.",
            "content": {"application/json": {"example": {
                "code": "MODEL_NOT_TRAINED",
                "message": "No trained churn model is available.",
                "details": [{"field": "model", "issue": "Call POST /model/train before requesting model status."}],
            }}},
        },
    },
)
def model_status():
    return model_service.get_model_status()


@router.get("/model/schema")
def model_schema():
    """Return the supported churn prediction features and their value constraints."""
    return model_service.get_feature_schema()


@router.get("/model/metrics")
def model_metrics(
    limit: int = Query(default=20, ge=1, le=100),
    model_type: str | None = Query(default=None, pattern="^(logreg|random_forest)$"),
):
    """Return the latest held-out metrics and recent successful training runs."""
    return model_service.get_training_metrics(limit=limit, model_type=model_type)
