"""
FraudLens AI - Prediction Module

Loads the trained XGBoost model and generates fraud predictions
and risk scores for transactions.
"""

from pathlib import Path

import joblib
import pandas as pd


# -------------------------------------------------------------------
# Project paths
# -------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "xgboost_fraud_model.joblib"
)

FEATURE_LIST_PATH = (
    PROJECT_ROOT
    / "models"
    / "feature_columns.joblib"
)


# -------------------------------------------------------------------
# Load trained model
# -------------------------------------------------------------------

def load_model():
    """Load the trained XGBoost model."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"\nModel not found:\n{MODEL_PATH}\n\n"
            "Please run train_model.py first."
        )

    model = joblib.load(MODEL_PATH)

    print("XGBoost fraud model loaded successfully.")

    return model


# -------------------------------------------------------------------
# Load feature list
# -------------------------------------------------------------------

def load_feature_list():
    """Load the feature columns used during training."""

    if not FEATURE_LIST_PATH.exists():
        raise FileNotFoundError(
            f"\nFeature list not found:\n"
            f"{FEATURE_LIST_PATH}"
        )

    feature_columns = joblib.load(
        FEATURE_LIST_PATH
    )

    return feature_columns


# -------------------------------------------------------------------
# Generate predictions
# -------------------------------------------------------------------

def predict_transactions(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate fraud probabilities and predictions.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Feature-engineered transaction data.

    Returns
    -------
    pandas.DataFrame
        Dataframe containing fraud probability,
        risk score and prediction.
    """

    model = load_model()

    feature_columns = load_feature_list()

    missing_features = [
        feature
        for feature in feature_columns
        if feature not in dataframe.columns
    ]

    if missing_features:
        raise ValueError(
            "\nMissing features required by model:\n"
            + "\n".join(
                f"- {feature}"
                for feature in missing_features
            )
        )

    X = dataframe[
        feature_columns
    ].copy()

    # Handle invalid numerical values.
    X = X.replace(
        [float("inf"), float("-inf")],
        0,
    )

    X = X.fillna(0)

    # Generate fraud probability.
    fraud_probability = model.predict_proba(
        X
    )[:, 1]

    # Convert probability to percentage risk score.
    risk_score = fraud_probability * 100

    # Create prediction labels.
    prediction = (
        fraud_probability >= 0.5
    ).astype(int)

    result = dataframe.copy()

    result["fraud_probability"] = (
        fraud_probability
    )

    result["risk_score"] = risk_score

    result["fraud_prediction"] = prediction

    return result


# -------------------------------------------------------------------
# Risk classification
# -------------------------------------------------------------------

def classify_risk(risk_score: float) -> str:
    """
    Convert numerical risk score into a human-readable
    risk category.
    """

    if risk_score >= 80:
        return "CRITICAL"

    if risk_score >= 60:
        return "HIGH"

    if risk_score >= 30:
        return "MEDIUM"

    return "LOW"


# -------------------------------------------------------------------
# Add risk categories
# -------------------------------------------------------------------

def add_risk_categories(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Add human-readable risk categories."""

    dataframe = dataframe.copy()

    dataframe["risk_category"] = (
        dataframe["risk_score"]
        .apply(classify_risk)
    )

    return dataframe


# -------------------------------------------------------------------
# Main test
# -------------------------------------------------------------------

def main():
    """Run a prediction test using the feature dataset."""

    print("=" * 60)
    print("FraudLens AI - Prediction Engine")
    print("=" * 60)

    feature_data_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "paysim_features.csv"
    )

    if not feature_data_path.exists():
        raise FileNotFoundError(
            f"\nFeature dataset not found:\n"
            f"{feature_data_path}"
        )

    print("\nLoading feature dataset...")

    dataframe = pd.read_csv(
        feature_data_path
    )

    print(
        f"Transactions loaded: "
        f"{len(dataframe):,}"
    )

    # For the initial test, process the first
    # 10,000 transactions.
    sample = dataframe.head(10_000).copy()

    print(
        "\nGenerating predictions for "
        f"{len(sample):,} transactions..."
    )

    result = predict_transactions(
        sample
    )

    result = add_risk_categories(
        result
    )

    print("\nPrediction completed.")

    print("\nRisk distribution:")

    print(
        result["risk_category"]
        .value_counts()
    )

    print("\nHighest-risk transactions:")

    display_columns = [
        "step",
        "type",
        "amount",
        "nameOrig",
        "nameDest",
        "risk_score",
        "risk_category",
        "fraud_prediction",
    ]

    print(
        result[
            display_columns
        ]
        .sort_values(
            "risk_score",
            ascending=False,
        )
        .head(10)
        .to_string(index=False)
    )

    print("\nPrediction engine test completed.")


if __name__ == "__main__":
    main()