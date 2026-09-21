from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURE_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "paysim_features.csv"
)

MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "xgboost_fraud_model.joblib"
)

FEATURE_COLUMNS_FILE = (
    PROJECT_ROOT
    / "models"
    / "feature_columns.joblib"
)


def test_feature_dataset_exists():
    assert FEATURE_FILE.exists(), (
        "Feature dataset was not found."
    )


def test_feature_dataset_can_be_loaded():
    data = pd.read_csv(FEATURE_FILE, nrows=10)

    assert not data.empty
    assert "amount" in data.columns
    assert "isFraud" in data.columns


def test_model_exists():
    assert MODEL_FILE.exists(), (
        "XGBoost model was not found."
    )


def test_model_can_be_loaded():
    model = joblib.load(MODEL_FILE)

    assert model is not None
    assert hasattr(model, "predict")


def test_feature_columns_exist():
    assert FEATURE_COLUMNS_FILE.exists(), (
        "Feature columns file was not found."
    )


def test_feature_columns_can_be_loaded():
    feature_columns = joblib.load(
        FEATURE_COLUMNS_FILE
    )

    assert isinstance(feature_columns, list)
    assert len(feature_columns) > 0