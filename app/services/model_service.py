from pathlib import Path
from typing import List, Optional, Tuple
from sklearn.base import RegressorMixin
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import cross_val_score
from statistics import mean


import joblib
import pandas as pd
import uuid

from app.dtos import Applicant, ModelMetadata, PredictionResult, TrainResult

score_funcs = {
    "linear": ["f_regression", "mutual_info_regression"],
    "logistic": ["f_classif", "mutual_info_classif", "chi2"],
}


class ModelService:
    @staticmethod
    def get_model(model_id: int) -> Optional[ModelMetadata]:
        return ModelMetadata(
            model_class="logistic",
            score_func="f_classif",
            num_features=10,
            learning_rate=0.1,
            k=2,
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
        X, y = ModelService._prepare_dataset(
            model_metadata.model_class,
            model_metadata.score_func,
            model_metadata.num_features,
        )

        if model_metadata.model_class == "linear":
            trained_model = LinearRegression().fit(X,y)
            train_predicted = trained_model.predict(X)
            train_accuracy = r2_score(y, train_predicted)

        elif model_metadata.model_class == "logistic":
            trained_model = LogisticRegression(random_state=0).fit(X,y)
            train_predicted = trained_model.predict(X)
            train_accuracy = accuracy_score(y, train_predicted)

        model_id = uuid.uuid4()
        validation_accuracy = mean(cross_val_score(trained_model,X,y,cv=model_metadata.k))

        ModelService._save_model(model_id, trained_model)
        ModelService._save_model_metadata(model_id, model_metadata)

        return TrainResult(
            model_id=model_id,
            train_acc=train_accuracy,
            valid_acc=validation_accuracy,
            )

    @staticmethod
    def predict(model_id: int, applicant: Applicant) -> PredictionResult:
        # TODO (kyungmin): Implement this function
        return PredictionResult(model_id=model_id, success=False)

    @staticmethod
    def _prepare_dataset(
        model_class: str, score_func: str, k: int
    ) -> Tuple[pd.DataFrame, pd.Series]:
        if model_class not in score_funcs.keys():
            raise ValueError(f"Unsupported model class: {model_class}")
        if score_func not in score_funcs[model_class]:
            raise ValueError(
                f"{model_class} model should use one of: {score_funcs[model_class]}"
            )

        data_dir = Path().cwd().parent.joinpath("data")
        with open(data_dir.joinpath(f"ranked-features-{score_func}.txt")) as f:
            features = [line.strip() for line in f.readlines()][:k]

        df = pd.read_csv(data_dir.joinpath("student-mat-preprocessed.csv"), sep=";")
        X, y = df.loc[:, df.columns.isin(features)], None
        if model_class == "linear":
            y = df["G3"]
        elif model_class == "logistic":
            y = df["G3"] >= 15.0
        return X, y
    
    @staticmethod
    def _save_model_metadata(model_id: str, model_metadata: ModelMetadata) -> None:
        filepath = f"../data/models/model_metadata/{model_metadata.model_class}_{model_id}.txt"

        with open(filepath,"w+") as f:
            f.write(f"Model Class : {model_metadata.model_class}\n")
            f.write(f"Score Function : {model_metadata.score_func}\n")
            f.write(f"Number of Features : {model_metadata.num_features}\n")
            f.write(f"Learning Rate : {model_metadata.learning_rate}\n")
            f.write(f"K : {model_metadata.k}\n")
    
        f.close()


    @staticmethod
    def _save_model(model_id: str, model: RegressorMixin) -> None:
        filepath = f"../data/models/model/model_{model_id}.pkl"

        with open(filepath, 'w+'):
            joblib.dump(model,filepath)
