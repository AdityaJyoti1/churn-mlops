"""
Inference API for Cloud Run.
Loads model from GCS at startup and serves /predict.
"""
import json
import os
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from google.cloud import storage


app = FastAPI(title="Churn Prediction API", version="1.0.0")

MODEL_BUCKET = os.getenv("MODEL_BUCKET")
MODEL_PREFIX = os.getenv("MODEL_PREFIX", "models/latest")

_state: Dict[str, Any] = {"model": None, "scaler": None, "features": None}


def load_artifact(blob_name: str, local_path: str) -> str:
    client = storage.Client()
    bucket = client.bucket(MODEL_BUCKET)
    bucket.blob(blob_name).download_to_filename(local_path)
    return local_path


@app.on_event("startup")
def load_model() -> None:
    """Pull artifacts from GCS once when the container starts."""
    if not MODEL_BUCKET:
        raise RuntimeError("MODEL_BUCKET env var not set")
    os.makedirs("/tmp/model", exist_ok=True)
    load_artifact(f"{MODEL_PREFIX}/model.joblib", "/tmp/model/model.joblib")
    load_artifact(f"{MODEL_PREFIX}/scaler.joblib", "/tmp/model/scaler.joblib")
    load_artifact(f"{MODEL_PREFIX}/features.json", "/tmp/model/features.json")

    _state["model"] = joblib.load("/tmp/model/model.joblib")
    _state["scaler"] = joblib.load("/tmp/model/scaler.joblib")
    with open("/tmp/model/features.json") as f:
        _state["features"] = json.load(f)
    print(f"Model loaded with {len(_state['features'])} features")


class PredictionRequest(BaseModel):
    instances: List[Dict[str, Any]]


class PredictionResponse(BaseModel):
    predictions: List[int]
    probabilities: List[float]
    model_version: str


@app.get("/")
def root():
    return {"service": "churn-prediction", "status": "ok"}


@app.get("/health")
def health():
    if _state["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "n_features": len(_state["features"])}


@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest):
    if _state["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    df = pd.DataFrame(req.instances)
    # Ensure all expected features are present, fill missing with 0
    for col in _state["features"]:
        if col not in df.columns:
            df[col] = 0
    df = df[_state["features"]]

    X = _state["scaler"].transform(df.values)
    preds = _state["model"].predict(X)
    probas = _state["model"].predict_proba(X)[:, 1]

    return PredictionResponse(
        predictions=preds.tolist(),
        probabilities=probas.tolist(),
        model_version=os.getenv("MODEL_VERSION", "latest"),
    )
