import pandas as pd
from sklearn.preprocessing import OrdinalEncoder


def preprocess() -> None:
    print("Preprocessing the dataset for logistic regression...", end="")
    df = pd.read_csv("student-mat.csv", sep=";")
    oe = OrdinalEncoder()
    ordinal_columns_set = {
        "school",
        "sex",
        "address",
        "famsize",
        "Pstatus",
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
    }
    ordinal_columns = list(ordinal_columns_set)
    logistic_df = pd.DataFrame(
        data=oe.fit_transform(df[ordinal_columns]), columns=ordinal_columns
    )
    for column in df.columns:
        if column not in ordinal_columns_set:
            logistic_df[column] = df[column]
    logistic_df.to_csv(path_or_buf="student-mat-logistic.csv", sep=";", index=False)
    print("DONE!")


if __name__ == "__main__":
    preprocess()
