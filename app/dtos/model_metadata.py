from dataclasses import dataclass
from typing import Optional

from flask_restx import fields


@dataclass(frozen=True)
class ModelMetadata:
    model_class: str
    num_features: int
    learning_rate: float
    k: Optional[int] = None


@dataclass(frozen=True)
class ModelMetadataFields:
    model_class: fields.String = fields.String(
        title="Model class",
        description="The name of the model class",
        enum=["logistic", "linear"],
        required=True,
    )
    num_features: fields.Integer = fields.Integer(
        title="Number of features",
        description="The number of features to select for training",
        min=1,
        max=51,
        required=True,
    )
    learning_rate: fields.Float = fields.Float(
        title="Learning rate",
        description="The learning rate to train a model with",
        required=True,
    )
    k: fields.Integer = fields.Integer(
        title="K-fold cross validation",
        description="Value used in K-fold cross validation",
        min=1,
    )
