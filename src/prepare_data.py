"""
Data preparation: download raw churn dataset, clean, encode, split, upload to GCS.
"""
import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from google.cloud import storage


CATEGORICAL_COLS = ["gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
                    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
                    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
                    "PaperlessBilling", "PaymentMethod"]
NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
TARGET_COL = "Churn"


def upload(bucket_name: str, blob_name: str, local_path: str) -> None:
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    bucket.blob(blob_name).upload_from_filename(local_path)
    print(f"Uploaded {local_path} -> gs://{bucket_name}/{blob_name}")


def prepare(args) -> None:
    df = pd.read_csv(args.input_csv)
    print(f"Loaded {len(df)} rows")

    # Drop ID column if present
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # TotalCharges has blanks for new customers — coerce and fill
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    # Encode target
    df["churn"] = (df[TARGET_COL] == "Yes").astype(int)
    df = df.drop(columns=[TARGET_COL])

    # One-hot encode categoricals
    df = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=True)

    # Cast booleans to int (one-hot output)
    for col in df.columns:
        if df[col].dtype == bool:
            df[col] = df[col].astype(int)

    # Train/test split
    train_df, test_df = train_test_split(
        df, test_size=0.2, stratify=df["churn"], random_state=42
    )
    print(f"Train: {len(train_df)}, Test: {len(test_df)}")

    os.makedirs("/tmp/processed", exist_ok=True)
    train_path = "/tmp/processed/train.csv"
    test_path = "/tmp/processed/test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    upload(args.bucket, "processed/train.csv", train_path)
    upload(args.bucket, "processed/test.csv", test_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True,
                        help="Path to raw churn CSV (Telco dataset)")
    parser.add_argument("--bucket", required=True)
    args = parser.parse_args()
    prepare(args)
