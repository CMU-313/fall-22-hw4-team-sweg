from dataclasses import dataclass

from flask_restx import fields


@dataclass(frozen=True)
class ModelMetadata:
    model_class: fields.String = fields.String(
        title="Model class",
        description="The name of the model class",
        enum=["logistic", "linear"],
        required=True,
    )
    learning_rate: fields.Float = fields.Float(
        title="Predicted success",
        description="The success of the given applicant predicted by the model",
        required=True,
    )
    k: fields.Integer = fields.Integer(
        title="K-fold cross validation",
        description="Value used in K-fold cross validation",
        min=1,
    )