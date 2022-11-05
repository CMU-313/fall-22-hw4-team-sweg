from typing import Callable

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_selection import (chi2, f_classif, f_regression,
                                       mutual_info_classif,
                                       mutual_info_regression)
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

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


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    oe = OrdinalEncoder()
    oe.fit(df[category_columns])
    joblib.dump(oe, "ordinal-encoder.pkl")
    ordinal_df = oe.transform(df[category_columns])

    ohe = OneHotEncoder(drop="if_binary", sparse=False)
    ohe.fit(ordinal_df)
    joblib.dump(ohe, "one-hot-encoder.pkl")
    one_hot_df = pd.DataFrame(
        data=ohe.transform(ordinal_df),
        columns=ohe.get_feature_names_out(input_features=category_columns),
    )
    category_column_set = set(category_columns)
    for column in df.columns:
        if column not in category_column_set:
            one_hot_df[column] = df[column]
    return one_hot_df


def rank_features():
    df = pd.read_csv("student-mat-preprocessed.csv", sep=";")
    X, y = df.loc[:, ~df.columns.isin(["G1", "G2", "G3"])], df["G3"]
    y_labels = y >= 15.0

    def rank_by_score_func(score_func: Callable, classifier: bool = True) -> None:
        scores = score_func(X, y_labels if classifier else y)
        if type(scores) == tuple:
            scores = scores[0]
        indices = np.argsort(scores)[::-1]
        with open(f"ranked-features-{score_func.__name__}.txt", "w") as f:
            for feature in X.columns[indices]:
                f.write(f"{feature}\n")

    for func in [f_regression, mutual_info_regression]:
        rank_by_score_func(func, classifier=False)
    for func in [f_classif, mutual_info_classif, chi2]:
        rank_by_score_func(func)


if __name__ == "__main__":
    df = pd.read_csv("student-mat.csv", sep=";")
    df = preprocess(df)
    df.to_csv(path_or_buf="student-mat-preprocessed.csv", sep=";", index=False)
    rank_features()
