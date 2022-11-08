from dataclasses import dataclass

from flask_restx import fields


@dataclass(frozen=True)
class PredictionResult:
    model_id: str
    success: bool


@dataclass(frozen=True)
class PredictionResultFields:
    model_id: fields.String = fields.String(
        title="Model ID",
        description="The ID of the model used to make prediction",
        required=True,
    )
    success: fields.Boolean = fields.Boolean(
        title="Predicted success",
        description="The success of the given applicant predicted by the model",
        required=True,
    )
