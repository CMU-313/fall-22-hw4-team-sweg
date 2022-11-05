from dataclasses import asdict
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

from app.dtos import Applicant, ModelMetadata, PredictionResult, TrainResult

category_columns = [
    "school",
    "sex",
    "address",
    "family_size",
    "p_status",
    "mother_edu",
    "father_edu",
    "mother_job",
    "father_job",
    "reason",
    "guardian",
    "school_support",
    "family_support",
    "paid",
    "activities",
    "nursery",
    "higher",
    "internet",
    "romantic",
]

score_funcs = {
    "linear": ["f_regression", "mutual_info_regression"],
    "logistic": ["f_classif", "mutual_info_classif", "chi2"],
}


class ModelService:
    @staticmethod
    def get_model(model_id: str) -> Optional[ModelMetadata]:
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

        # TODO (jaehoon): Split dataset and train a model

        # TODO (jaehoon): Export the trained model and the metadata into pkl and json files, respectively
        return TrainResult(model_id=1, train_acc=0.5, valid_acc=0.5)

    @staticmethod
    def predict(model_id: str, model_metadata: ModelMetadata, applicant: Applicant) -> PredictionResult:
        data_dir = Path().cwd().parent.joinpath("data")
        with open(data_dir.joinpath(f"ranked-features-{model_metadata.score_func}.txt")) as f:
            features = [line.strip() for line in f.readlines()][:model_metadata.num_features]

        df = pd.DataFrame.from_dict(asdict(applicant))
        oe = OrdinalEncoder()
        ohe = OneHotEncoder(drop="if_binary", sparse=False)
        one_hot_df = pd.DataFrame(
            data=ohe.fit_transform(oe.fit_transform(df[category_columns])),
            columns=ohe.get_feature_names_out(input_features=category_columns),
        )
        category_column_set = set(category_columns)
        for column in df.columns:
            if column not in category_column_set:
                one_hot_df[column] = df[column]
        print(one_hot_df)
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
