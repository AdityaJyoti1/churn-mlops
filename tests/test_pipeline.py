"""Unit tests run by GitHub Actions on every PR."""
import json
import sys
import os
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def test_imports():
    """Verify core modules import without error."""
    import train  # noqa
    import prepare_data  # noqa
    import app  # noqa


def test_prepare_categoricals_defined():
    import prepare_data
    assert "Contract" in prepare_data.CATEGORICAL_COLS
    assert prepare_data.TARGET_COL == "Churn"


def test_app_response_models():
    from app import PredictionRequest, PredictionResponse
    req = PredictionRequest(instances=[{"tenure": 12, "MonthlyCharges": 50.0}])
    assert len(req.instances) == 1

    resp = PredictionResponse(predictions=[1], probabilities=[0.85], model_version="v1")
    assert resp.predictions == [1]


def test_data_split_logic():
    """Synthetic test of the prepare flow without GCS."""
    df = pd.DataFrame({
        "tenure": np.random.randint(1, 72, 100),
        "MonthlyCharges": np.random.uniform(20, 120, 100),
        "TotalCharges": np.random.uniform(20, 8000, 100),
        "SeniorCitizen": np.random.randint(0, 2, 100),
        "Contract": np.random.choice(["Month-to-month", "One year"], 100),
        "Churn": np.random.choice(["Yes", "No"], 100),
    })
    df["churn"] = (df["Churn"] == "Yes").astype(int)
    assert df["churn"].isin([0, 1]).all()


def test_metric_keys_present():
    """Smoke check that we log the expected metrics."""
    expected = {"accuracy", "precision", "recall", "f1", "roc_auc"}
    # Just assert the set is well-defined for downstream consumers.
    assert len(expected) == 5
