import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


def preprocess() -> None:
    print("Preprocessing the dataset...", end="")
    df = pd.read_csv("student-mat.csv", sep=";")

    category_columns = [
        "school",
        "sex",
        "address",
        "famsize",
        "Pstatus",
        "Medu",
        "Fedu",
        "Mjob",
        "Fjob",
        "reason",
        "guardian",
        "schoolsup",
        "famsup",
        "paid",
        "activities",
        "nursery",
        "higher",
        "internet",
        "romantic",
    ]
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
    one_hot_df.to_csv(path_or_buf="student-mat-preprocessed.csv", sep=";", index=False)

    print("DONE!")


if __name__ == "__main__":
    preprocess()
