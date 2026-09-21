"""
FraudLens AI - Model Training Module

Trains and compares:
1. Logistic Regression
2. XGBoost

The best model is selected using ROC-AUC and saved for prediction.
"""

from pathlib import Path

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)

from xgboost import XGBClassifier


# -------------------------------------------------------------------
# Project paths
# -------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "paysim_features.csv"
)

MODEL_DIR = PROJECT_ROOT / "models"

LOGISTIC_MODEL_PATH = (
    MODEL_DIR / "logistic_regression.joblib"
)

XGBOOST_MODEL_PATH = (
    MODEL_DIR / "xgboost_fraud_model.joblib"
)

SCALER_PATH = (
    MODEL_DIR / "feature_scaler.joblib"
)

FEATURE_LIST_PATH = (
    MODEL_DIR / "feature_columns.joblib"
)


# -------------------------------------------------------------------
# Features used for machine learning
# -------------------------------------------------------------------

FEATURE_COLUMNS = [
    "step",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "isFlaggedFraud",
    "type_encoded",
    "origin_balance_change",
    "destination_balance_change",
    "origin_balance_error",
    "destination_balance_error",
    "amount_to_origin_balance",
    "amount_to_destination_balance",
    "origin_zero_balance",
    "destination_zero_balance",
    "large_transaction",
    "log_amount",
    "hour",
    "day",
    "is_night",
    "same_origin_destination",
]


# -------------------------------------------------------------------
# Load data
# -------------------------------------------------------------------

def load_data() -> pd.DataFrame:
    """Load the feature-engineered dataset."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"\nFeature dataset not found:\n{DATA_PATH}\n\n"
            "Please run feature_engineering.py first."
        )

    print(f"Loading dataset:\n{DATA_PATH}")

    dataframe = pd.read_csv(DATA_PATH)

    print("\nDataset loaded successfully.")
    print(f"Rows    : {len(dataframe):,}")
    print(f"Columns : {len(dataframe.columns)}")

    return dataframe


# -------------------------------------------------------------------
# Prepare ML data
# -------------------------------------------------------------------

def prepare_data(dataframe: pd.DataFrame):
    """Prepare features and target."""

    missing_features = [
        feature
        for feature in FEATURE_COLUMNS
        if feature not in dataframe.columns
    ]

    if missing_features:
        raise ValueError(
            "\nMissing required features:\n"
            + "\n".join(
                f"- {feature}"
                for feature in missing_features
            )
        )

    X = dataframe[FEATURE_COLUMNS].copy()

    y = dataframe["isFraud"].astype(int)

    # Replace invalid numerical values.
    X = X.replace(
        [float("inf"), float("-inf")],
        0,
    )

    X = X.fillna(0)

    print("\nFeature matrix prepared.")
    print(f"Features : {X.shape[1]}")
    print(f"Fraud cases : {int(y.sum()):,}")
    print(f"Legitimate cases : {int((y == 0).sum()):,}")

    return X, y


# -------------------------------------------------------------------
# Train-test split
# -------------------------------------------------------------------

def split_data(X, y):
    """Create stratified training and testing datasets."""

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print("\nData split completed.")

    print(f"Training samples : {len(X_train):,}")
    print(f"Testing samples  : {len(X_test):,}")

    return X_train, X_test, y_train, y_test


# -------------------------------------------------------------------
# Train Logistic Regression
# -------------------------------------------------------------------

def train_logistic_regression(
    X_train,
    y_train,
    X_test,
    y_test,
):
    """Train the baseline Logistic Regression model."""

    print("\n" + "=" * 60)
    print("TRAINING LOGISTIC REGRESSION")
    print("=" * 60)

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )

    model.fit(
        X_train_scaled,
        y_train,
    )

    probabilities = model.predict_proba(
        X_test_scaled
    )[:, 1]

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    roc_auc = roc_auc_score(
        y_test,
        probabilities,
    )

    average_precision = average_precision_score(
        y_test,
        probabilities,
    )

    print(f"\nROC-AUC            : {roc_auc:.4f}")
    print(
        f"Average Precision : {average_precision:.4f}"
    )

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            digits=4,
        )
    )

    print("Confusion Matrix:")
    print(
        confusion_matrix(
            y_test,
            predictions,
        )
    )

    return (
        model,
        scaler,
        roc_auc,
        average_precision,
    )


# -------------------------------------------------------------------
# Train XGBoost
# -------------------------------------------------------------------

def train_xgboost(
    X_train,
    y_train,
    X_test,
    y_test,
):
    """Train the main XGBoost fraud detection model."""

    print("\n" + "=" * 60)
    print("TRAINING XGBOOST")
    print("=" * 60)

    fraud_count = int(y_train.sum())

    legitimate_count = int(
        (y_train == 0).sum()
    )

    scale_pos_weight = (
        legitimate_count / fraud_count
        if fraud_count > 0
        else 1
    )

    print(
        f"\nCalculated scale_pos_weight: "
        f"{scale_pos_weight:.2f}"
    )

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="auc",
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
    )

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    roc_auc = roc_auc_score(
        y_test,
        probabilities,
    )

    average_precision = average_precision_score(
        y_test,
        probabilities,
    )

    print(f"\nROC-AUC            : {roc_auc:.4f}")
    print(
        f"Average Precision : {average_precision:.4f}"
    )

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            digits=4,
    ))

    print("Confusion Matrix:")
    print(
        confusion_matrix(
            y_test,
            predictions,
        )
    )

    return (
        model,
        roc_auc,
        average_precision,
    )


# -------------------------------------------------------------------
# Save models
# -------------------------------------------------------------------

def save_models(
    logistic_model,
    scaler,
    xgb_model,
):
    """Save trained models and metadata."""

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        logistic_model,
        LOGISTIC_MODEL_PATH,
    )

    joblib.dump(
        scaler,
        SCALER_PATH,
    )

    joblib.dump(
        xgb_model,
        XGBOOST_MODEL_PATH,
    )

    joblib.dump(
        FEATURE_COLUMNS,
        FEATURE_LIST_PATH,
    )

    print("\nModels saved successfully.")

    print(
        f"Logistic Regression:\n"
        f"{LOGISTIC_MODEL_PATH}"
    )

    print(
        f"\nScaler:\n"
        f"{SCALER_PATH}"
    )

    print(
        f"\nXGBoost:\n"
        f"{XGBOOST_MODEL_PATH}"
    )

    print(
        f"\nFeature list:\n"
        f"{FEATURE_LIST_PATH}"
    )


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main():
    """Run complete model training pipeline."""

    print("=" * 60)
    print("FraudLens AI - Model Training")
    print("=" * 60)

    dataframe = load_data()

    X, y = prepare_data(dataframe)

    X_train, X_test, y_train, y_test = split_data(
        X,
        y,
    )

    (
        logistic_model,
        scaler,
        logistic_auc,
        logistic_ap,
    ) = train_logistic_regression(
        X_train,
        y_train,
        X_test,
        y_test,
    )

    (
        xgb_model,
        xgb_auc,
        xgb_ap,
    ) = train_xgboost(
        X_train,
        y_train,
        X_test,
        y_test,
    )

    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    print(
        f"\nLogistic Regression"
        f"\n  ROC-AUC            : {logistic_auc:.4f}"
        f"\n  Average Precision : {logistic_ap:.4f}"
    )

    print(
        f"\nXGBoost"
        f"\n  ROC-AUC            : {xgb_auc:.4f}"
        f"\n  Average Precision : {xgb_ap:.4f}"
    )

    # Save both models.
    save_models(
        logistic_model,
        scaler,
        xgb_model,
    )

    print("\n" + "=" * 60)
    print("MODEL TRAINING COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()