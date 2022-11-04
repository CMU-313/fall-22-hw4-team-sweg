from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif, f_regression

from app.dtos import Applicant, ModelMetadata, PredictionResult, TrainResult


class ModelService:
    @staticmethod
    def get_model(model_id: int) -> Optional[ModelMetadata]:
        return ModelMetadata(
            model_class="logistic", num_features=10, learning_rate=0.1, k=2
        )

    @staticmethod
    def get_model_list() -> List[ModelMetadata]:
        # empty list so far, but should fetch the data from a
        # specific directory later, and parse out the data into
        # a list
        return []

    @staticmethod
    def delete(model_id: int) -> None:
        # TODO (victor): Implement this function
        return None

    @staticmethod
    def train(model_metadata: ModelMetadata) -> TrainResult:
        X, y = ModelService.select_features(
            model_metadata.model_class, model_metadata.num_features
        )

        # TODO (jaehoon): Split dataset and train a model

        # TODO (jaehoon): Export the trained model and the metadata into pkl and json files, respectively
        return TrainResult(model_id=1, train_acc=0.5, valid_acc=0.5)

    @staticmethod
    def predict(model_id: int, applicant: Applicant) -> PredictionResult:
        # TODO (kyungmin): Implement this function
        return PredictionResult(model_id=model_id, success=False)

    @staticmethod
    def select_features(model_class: str, k: int) -> Tuple[pd.DataFrame, pd.Series]:
        if model_class not in ["linear", "logistic"]:
            raise ValueError(f"Unsupported model class: {model_class}")

        model_class = "linear"
        file_path = Path().cwd().parent.joinpath(f"data/student-mat-preprocessed.csv")
        df = pd.read_csv(file_path, sep=";")
        X, y = df.loc[:, ~df.columns.isin(["G1", "G2", "G3"])], None
        score_func = None
        if model_class == "linear":
            y = df["G3"]
            score_func = f_regression
        elif model_class == "logistic":
            y = df["G3"] >= 15.0
            score_func = f_classif
        selector = SelectKBest(score_func=score_func, k=min(len(X.columns), k))
        return pd.DataFrame(data=selector.fit_transform(X, y=y)), y
