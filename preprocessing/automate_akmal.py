import pandas as pd
from sklearn.preprocessing import StandardScaler
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

input_file = (
    BASE_DIR.parent
    / "dataset-raw"
    / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

output_file = (
    BASE_DIR
    / "customer_churn_processed.csv"
)

def preprocess_data(
        input_path,
        output_path
):
    df = pd.read_csv(input_path)

    df = df.drop(
        columns=["customerID"]
    )

    df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
    )

    df["TotalCharges"] = df["TotalCharges"].fillna(
    df["TotalCharges"].median()
    )

    df["Churn"] = df["Churn"].map(
        {
            "No": 0,
            "Yes": 1
        }
    )

    categorical_cols = (
        df.select_dtypes(
            include="object"
        ).columns
    )

    df = pd.get_dummies(
        df,
        columns=categorical_cols,
        drop_first=True
    )

    scaler = StandardScaler()

    numerical_cols = [
        "tenure",
        "MonthlyCharges",
        "TotalCharges"
    ]

    df[numerical_cols] = scaler.fit_transform(
        df[numerical_cols]
    )

    df.to_csv(
        output_path,
        index=False
    )

    print(
        f"Processed dataset saved to {output_path}"
    )

if __name__ == "__main__":
    preprocess_data(
        input_file,
        output_file
    )