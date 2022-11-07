from dataclasses import dataclass

from flask_restx import fields


@dataclass(frozen=True)
class TrainMetadata:
    model_class: str
    score_func: str
    num_features: int
    k: int


@dataclass(frozen=True)
class TrainResult:
    model_id: str
    train_acc: float
    valid_acc: float


@dataclass(frozen=True)
class TrainMetadataFields:
    model_class: fields.String = fields.String(
        title="Model class",
        description="The name of the model class",
        enum=["logistic", "linear"],
        required=True,
    )
    score_func: fields.String = fields.String(
        title="Score function",
        description="The score function to use to select features",
        enum=[
            "f_regression",
            "mutual_info_regression",
            "f_classif",
            "mutual_info_classif",
            "chi2",
        ],
        required=True,
    )
    num_features: fields.Integer = fields.Integer(
        title="Number of features",
        description="The number of features to select for training",
        min=1,
        max=51,
        required=True,
    )
    k: fields.Integer = fields.Integer(
        title="K-fold cross validation",
        description="Value used in K-fold cross validation",
        min=2,
        required=True,
    )


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
