import os
import uuid
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from typing import List, Optional, Tuple

import joblib
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import cross_val_score

from app.dtos import Applicant, ModelMetadata, PredictionResult, TrainResult
from app.dtos.train import TrainMetadata
from data.preprocessor import preprocess

score_funcs = {
    "linear": ["f_regression", "mutual_info_regression"],
    "logistic": ["f_classif", "mutual_info_classif", "chi2"],
}

data_dir = Path().cwd().parent.joinpath("data")


class ModelService:
    @staticmethod
    def get_model(model_id: str) -> Optional[ModelMetadata]:
        try:
            f = open(f"{model_id}.txt", "r")
        except FileNotFoundError:
            return None

        data = f.readlines()
        f.close()
        return ModelMetadata(
            model_id=model_id,
            model_class=data[0].split(":")[1],
            score_func=data[1].split(":")[1],
            num_features=int(data[2].split(":")[1]),
            k=int(data[3].split(":")[1]),
            train_acc=float(data[4].split(":")[1]),
            valid_acc=float(data[5].split(":")[1]),
        )

    @staticmethod
    def get_model_list() -> List[ModelMetadata]:
        model_list = []
        for filename in os.listdir(data_dir.joinpath("models")):
            if not filename.endswith(".txt"):
                continue
            with open(filename, "r") as f:
                data = f.readlines()
            model_list.append(
                ModelMetadata(
                    model_id=filename[:-4],
                    model_class=data[0].split(":")[1].strip(),
                    score_func=data[1].split(":")[1].strip(),
                    num_features=int(data[2].split(":")[1]),
                    k=int(data[3].split(":")[1]),
                    train_acc=float(data[4].split(":")[1]),
                    valid_acc=float(data[5].split(":")[1]),
                )
            )
        return model_list

    @staticmethod
    def delete(model_id: str) -> None:
        # Delete model and its related metadata
        try:
            os.remove(f"models/{model_id}.txt")  # Delete model metadata
            os.remove(f"models/{model_id}.pkl")  # Delete model
        except OSError:
            return None

    @staticmethod
    def train(train_metadata: TrainMetadata) -> TrainResult:
        X, y = ModelService._prepare_dataset(
            train_metadata.model_class,
            train_metadata.score_func,
            train_metadata.num_features,
        )

        model, train_accuracy = None, 0.0
        if train_metadata.model_class == "linear":
            model = LinearRegression().fit(X, y)
            train_predicted = model.predict(X)
            train_accuracy = r2_score(y, train_predicted)
        elif train_metadata.model_class == "logistic":
            model = LogisticRegression(max_iter=1000).fit(X, y)
            train_predicted = model.predict(X)
            train_accuracy = accuracy_score(y, train_predicted)

        model_id = str(uuid.uuid4())
        validation_accuracy = mean(cross_val_score(model, X, y, cv=train_metadata.k))

        # Export the model
        ModelService._save_model(model_id, model)
        ModelService._save_model_metadata(
            ModelMetadata(
                **asdict(train_metadata),
                model_id=model_id,
                train_acc=train_accuracy,
                valid_acc=validation_accuracy,
            )
        )

        return TrainResult(
            model_id=model_id,
            train_acc=train_accuracy,
            valid_acc=validation_accuracy,
        )

    @staticmethod
    def predict(
        model_id: str, model_metadata: ModelMetadata, applicant: Applicant
    ) -> PredictionResult:
        df = pd.DataFrame(asdict(applicant), index=[0])
        X, _ = ModelService._prepare_dataset(
            model_metadata.model_class,
            model_metadata.score_func,
            model_metadata.num_features,
            df=preprocess(df, predict=True),
        )
        model = joblib.load(data_dir.joinpath(f"models/{model_id}.pkl"))
        out = model.predict(X)[0]
        if model_metadata.model_class == "linear":
            out = out >= 15.0
        elif model_metadata.model_class == "logistic":
            out = bool(out)
        return PredictionResult(model_id=model_id, success=out)

    @staticmethod
    def _prepare_dataset(
        model_class: str, score_func: str, k: int, df: Optional[pd.DataFrame] = None
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        if model_class not in score_funcs.keys():
            raise ValueError(f"Unsupported model class: {model_class}")
        if score_func not in score_funcs[model_class]:
            raise ValueError(
                f"{model_class} model should use one of: {score_funcs[model_class]}"
            )

        with open(data_dir.joinpath(f"features/ranked-features-{score_func}.txt")) as f:
            features = [line.strip() for line in f.readlines()][:k]

        if df is None:
            df = pd.read_csv(data_dir.joinpath("student-mat-preprocessed.csv"), sep=";")
        X, y = df.loc[:, df.columns.isin(features)], None
        if "G3" not in df.columns:
            return X, y
        if model_class == "linear":
            y = df["G3"]
        elif model_class == "logistic":
            y = df["G3"] >= 15.0
        return X, y

    @staticmethod
    def _save_model_metadata(model_metadata: ModelMetadata) -> None:
        model_dir = data_dir.joinpath(f"models/{model_metadata.model_id}.txt")
        with open(model_dir, "w") as f:
            f.write(f"Model Class:{model_metadata.model_class}\n")
            f.write(f"Score Function:{model_metadata.score_func}\n")
            f.write(f"Number of Features:{model_metadata.num_features}\n")
            f.write(f"K:{model_metadata.k}\n")
            f.write(f"Train Accuracy:{model_metadata.train_acc}\n")
            f.write(f"Validation Accuracy:{model_metadata.valid_acc}\n")

    @staticmethod
    def _save_model(model_id: str, model: RegressorMixin) -> None:
        model_dir = data_dir.joinpath(f"models/{model_id}.pkl")
        with open(model_dir, "w"):
            joblib.dump(model, model_dir)
