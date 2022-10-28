from dataclasses import dataclass

from flask_restx import fields


@dataclass(frozen=True)
class PredictionResult:
    model_id: fields.Integer = fields.Integer(
        title="Model ID",
        description="The ID of the model used to make prediction",
        required=True,
        min=1,
    )
    success: fields.Boolean = fields.Boolean(
        title="Predicted Success",
        description="The success of the given student predicted by the model",
        required=True,
    )
