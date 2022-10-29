from dataclasses import dataclass

from flask_restx import fields


@dataclass(frozen=True)
class TrainResultFields:
    model_id: fields.Integer = fields.Integer(
        title="Model ID",
        description="The ID of the created model",
        required=True,
        min=1,
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
