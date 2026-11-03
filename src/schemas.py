from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class TrainingConfigChurn(BaseModel):
    model_type: str = Field(default="logreg", description="Classifier: logreg or random_forest")
    hyperparameters: dict = Field(default_factory=dict, description="Supported classifier settings")

    @field_validator("model_type")
    @classmethod
    def validate_model_type(cls, value: str) -> str:
        if value not in {"logreg", "random_forest"}:
            raise ValueError("model_type must be 'logreg' or 'random_forest'")
        return value


class FeatureVectorChurn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    monthly_fee: float
    usage_hours: float
    support_requests: int
    account_age_months: int
    failed_payments: int
    region: Literal["europe", "asia", "america", "africa"]
    device_type: Literal["mobile", "desktop", "tablet"]
    payment_method: Literal["card", "paypal", "crypto"]
    autopay_enabled: Literal[0, 1] = Field(description="Binary value: 0 or 1")

class DatasetRowChurn(FeatureVectorChurn):
    churn: int

class PredictionResponseChurn(BaseModel):
    predicted_class: int = Field(description="Predicted churn class: 1 means churn, 0 means stayed")
    class_probabilities: dict[str, float] = Field(
        description="Probability for each churn class, keyed by class label (0 or 1)"
    )
