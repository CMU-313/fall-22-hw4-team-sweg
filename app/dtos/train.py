from dataclasses import dataclass

from flask_restx import fields


@dataclass(frozen=True)
class TrainResult:
    model_id: str
    train_acc: float
    valid_acc: float


@dataclass(frozen=True)
class TrainResultFields:
    model_id: fields.String = fields.String(
        title="Model ID",
        description="The ID of the created model",
        required=True,
    )
    train_acc: fields.Float = fields.Float(
        title="Training accuracy",
        description="Model accuracy tested on training set",
        required=True,
        min=0.0,
        max=1.0,
    )
    valid_acc: fields.Float = fields.Float(
        title="Validation accuracy",
        description="Model accuracy tested on validation set",
        min=0.0,
        max=1.0,
    )
