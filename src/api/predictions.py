from typing import Annotated
import logging

from fastapi import APIRouter, Body, HTTPException

from service import model as model_service
from schemas import FeatureVectorChurn, PredictionResponseChurn

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/")
def get_message():
    return {"message": "ml churn service is running"}

@router.post(
    "/predict",
    response_model=PredictionResponseChurn | list[PredictionResponseChurn],
    summary="Predict customer churn",
    description=(
        "Submit one customer feature object to receive one prediction object, or a list of "
        "feature objects to receive a list of predictions in the same order. Probabilities "
        "are keyed by churn class (`0` stayed, `1` churned)."
    ),
    responses={
        200: {
            "description": "One prediction object for a single input, or an ordered list for a batch.",
            "content": {
                "application/json": {
                    "examples": {
                        "single_prediction": {
                            "summary": "Single customer response",
                            "value": {
                                "predicted_class": 1,
                                "class_probabilities": {"0": 0.12, "1": 0.88},
                            },
                        },
                        "batch_predictions": {
                            "summary": "Batch response in request order",
                            "value": [
                                {"predicted_class": 1, "class_probabilities": {"0": 0.12, "1": 0.88}},
                                {"predicted_class": 0, "class_probabilities": {"0": 0.91, "1": 0.09}},
                            ],
                        },
                    }
                }
            },
        },
        422: {
            "description": "Feature fields are missing, have invalid types, or include unsupported fields.",
            "content": {"application/json": {"example": {
                "code": "VALIDATION_ERROR",
                "message": "Request data does not match the churn API schema.",
                "details": [{"field": "body.monthly_fee", "issue": "Field required"}],
            }}},
        },
        503: {
            "description": "No trained model is available.",
            "content": {"application/json": {"example": {
                "code": "MODEL_NOT_TRAINED", "message": "No trained churn model is available.",
                "details": [{"field": "model", "issue": "Call POST /model/train before predicting."}],
            }}},
        },
    },
)
def get_prediction(
    request: Annotated[
        FeatureVectorChurn | list[FeatureVectorChurn],
        Body(
            openapi_examples={
                "single_customer": {
                    "summary": "Single customer",
                    "value": {
                        "monthly_fee": 79.99, "usage_hours": 4.5,
                        "support_requests": 3, "account_age_months": 2,
                        "failed_payments": 2, "region": "europe",
                        "device_type": "mobile", "payment_method": "crypto",
                        "autopay_enabled": 0,
                    },
                },
                "batch_customers": {
                    "summary": "Batch of two customers",
                    "value": [
                        {
                            "monthly_fee": 79.99, "usage_hours": 4.5,
                            "support_requests": 3, "account_age_months": 2,
                            "failed_payments": 2, "region": "europe",
                            "device_type": "mobile", "payment_method": "crypto",
                            "autopay_enabled": 0,
                        },
                        {
                            "monthly_fee": 35.0, "usage_hours": 65.0,
                            "support_requests": 0, "account_age_months": 36,
                            "failed_payments": 0, "region": "asia",
                            "device_type": "desktop", "payment_method": "card",
                            "autopay_enabled": 1,
                        },
                    ],
                },
            }
        ),
    ],
) -> PredictionResponseChurn | list[PredictionResponseChurn]:
    is_batch = isinstance(request, list)
    if is_batch and not request:
        raise HTTPException(status_code=422, detail={
            "code": "VALIDATION_ERROR",
            "message": "Request data does not match the churn API schema.",
            "details": [{"field": "body", "issue": "Provide at least one customer feature object."}],
        })
    records = [item.model_dump() for item in request] if is_batch else [request.model_dump()]
    logger.info("Prediction request received: records=%d", len(records))
    predictions = model_service.predict_customers(records)
    if is_batch:
        return predictions
    return predictions[0]
