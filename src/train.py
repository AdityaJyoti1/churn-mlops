"""
Customer Churn Prediction - Training Pipeline
Logs experiments to MLflow tracking server running on Compute Engine VM.
"""
import os
import argparse
import json
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from google.cloud import storage
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)


def download_from_gcs(bucket_name: str, blob_name: str, local_path: str) -> None:
    """Download a file from GCS to local disk."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.download_to_filename(local_path)
    print(f"Downloaded gs://{bucket_name}/{blob_name} -> {local_path}")


def upload_to_gcs(bucket_name: str, blob_name: str, local_path: str) -> None:
    """Upload a file to GCS."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(local_path)
    print(f"Uploaded {local_path} -> gs://{bucket_name}/{blob_name}")


def load_data(bucket: str, train_blob: str, test_blob: str) -> tuple:
    """Load processed train/test data from GCS."""
    os.makedirs("/tmp/data", exist_ok=True)
    download_from_gcs(bucket, train_blob, "/tmp/data/train.csv")
    download_from_gcs(bucket, test_blob, "/tmp/data/test.csv")
    train_df = pd.read_csv("/tmp/data/train.csv")
    test_df = pd.read_csv("/tmp/data/test.csv")
    return train_df, test_df


def train_model(args) -> None:
    """End-to-end training with MLflow tracking."""
    # Configure MLflow
    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(args.experiment_name)

    with mlflow.start_run() as run:
        print(f"MLflow run_id: {run.info.run_id}")

        # ---- Load data ----
        train_df, test_df = load_data(args.bucket, args.train_blob, args.test_blob)
        feature_cols = [c for c in train_df.columns if c != "churn"]
        X_train, y_train = train_df[feature_cols], train_df["churn"]
        X_test, y_test = test_df[feature_cols], test_df["churn"]

        # ---- Scale ----
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        # ---- Log params ----
        params = {
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "min_samples_split": args.min_samples_split,
            "random_state": 42,
            "n_features": len(feature_cols),
            "n_train": len(X_train),
            "n_test": len(X_test),
        }
        mlflow.log_params(params)

        # ---- Train ----
        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_split=args.min_samples_split,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train_s, y_train)

        # ---- Evaluate ----
        y_pred = model.predict(X_test_s)
        y_proba = model.predict_proba(X_test_s)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba),
        }
        mlflow.log_metrics(metrics)
        print("Metrics:", json.dumps(metrics, indent=2))

        # ---- Persist artifacts locally ----
        os.makedirs("/tmp/artifacts", exist_ok=True)
        model_path = "/tmp/artifacts/model.joblib"
        scaler_path = "/tmp/artifacts/scaler.joblib"
        features_path = "/tmp/artifacts/features.json"
        cm_path = "/tmp/artifacts/confusion_matrix.json"

        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
        with open(features_path, "w") as f:
            json.dump(feature_cols, f)
        with open(cm_path, "w") as f:
            json.dump(confusion_matrix(y_test, y_pred).tolist(), f)

        # ---- Log to MLflow ----
        mlflow.sklearn.log_model(model, "model", registered_model_name="churn_classifier")
        mlflow.log_artifact(scaler_path, artifact_path="preprocessing")
        mlflow.log_artifact(features_path, artifact_path="metadata")
        mlflow.log_artifact(cm_path, artifact_path="metrics")

        # ---- Upload to GCS for serving ----
        upload_to_gcs(args.bucket, "models/latest/model.joblib", model_path)
        upload_to_gcs(args.bucket, "models/latest/scaler.joblib", scaler_path)
        upload_to_gcs(args.bucket, "models/latest/features.json", features_path)

        # Tag run for traceability
        mlflow.set_tag("deployment_target", "cloud_run")
        mlflow.set_tag("git_commit", os.getenv("GITHUB_SHA", "local"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--train-blob", default="processed/train.csv")
    parser.add_argument("--test-blob", default="processed/test.csv")
    parser.add_argument("--mlflow-uri", required=True,
                        help="MLflow tracking URI, e.g. http://VM_IP:5000")
    parser.add_argument("--experiment-name", default="churn-prediction")
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=10)
    parser.add_argument("--min-samples-split", type=int, default=2)
    args = parser.parse_args()
    train_model(args)
