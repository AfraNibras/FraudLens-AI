"""
FraudLens AI - Explainable AI Module

Uses SHAP to explain XGBoost fraud predictions.
"""

from pathlib import Path

import joblib
import pandas as pd
import shap


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
# Load model
# -------------------------------------------------------------------

def load_model():
    """Load the trained XGBoost model."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"\nModel not found:\n{MODEL_PATH}\n\n"
            "Run train_model.py first."
        )

    return joblib.load(MODEL_PATH)


# -------------------------------------------------------------------
# Load feature list
# -------------------------------------------------------------------

def load_feature_list():
    """Load the feature list used during training."""

    if not FEATURE_LIST_PATH.exists():
        raise FileNotFoundError(
            f"\nFeature list not found:\n"
            f"{FEATURE_LIST_PATH}"
        )

    return joblib.load(
        FEATURE_LIST_PATH
    )


# -------------------------------------------------------------------
# Prepare transaction
# -------------------------------------------------------------------

def prepare_features(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Prepare transaction features for SHAP."""

    feature_columns = load_feature_list()

    missing_features = [
        feature
        for feature in feature_columns
        if feature not in dataframe.columns
    ]

    if missing_features:
        raise ValueError(
            "\nMissing features:\n"
            + "\n".join(
                f"- {feature}"
                for feature in missing_features
            )
        )

    X = dataframe[
        feature_columns
    ].copy()

    X = X.replace(
        [float("inf"), float("-inf")],
        0,
    )

    X = X.fillna(0)

    return X


# -------------------------------------------------------------------
# Create SHAP explainer
# -------------------------------------------------------------------

def create_explainer():
    """Create a SHAP TreeExplainer."""

    model = load_model()

    explainer = shap.TreeExplainer(
        model
    )

    return explainer


# -------------------------------------------------------------------
# Explain transactions
# -------------------------------------------------------------------

def explain_transactions(
    dataframe: pd.DataFrame,
):
    """
    Generate SHAP explanations for transactions.

    Returns
    -------
    shap.Explanation
        SHAP values for the supplied transactions.
    """

    X = prepare_features(
        dataframe
    )

    explainer = create_explainer()

    shap_values = explainer(
        X
    )

    return shap_values


# -------------------------------------------------------------------
# Get feature contributions
# -------------------------------------------------------------------

def get_feature_contributions(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Return feature contributions for supplied transactions.

    Positive values generally push the prediction toward
    the fraud class, while negative values push it away.
    """

    X = prepare_features(
        dataframe
    )

    shap_values = explain_transactions(
        dataframe
    )

    values = shap_values.values

    # For binary classification, SHAP may return either:
    # - 2D array: rows x features
    # - 3D array: rows x features x classes
    if values.ndim == 3:
        values = values[:, :, 1]

    contribution_dataframe = pd.DataFrame(
        values,
        columns=X.columns,
        index=X.index,
    )

    return contribution_dataframe


# -------------------------------------------------------------------
# Explain one transaction
# -------------------------------------------------------------------

def explain_single_transaction(
    dataframe: pd.DataFrame,
    transaction_index: int = 0,
):
    """
    Explain one transaction and return its most important
    contributing features.
    """

    if transaction_index >= len(dataframe):
        raise IndexError(
            "Transaction index is outside the supplied dataset."
        )

    transaction = dataframe.iloc[
        [transaction_index]
    ].copy()

    contributions = get_feature_contributions(
        transaction
    )

    contribution_series = (
        contributions.iloc[0]
        .sort_values(
            key=abs,
            ascending=False,
        )
    )

    explanation = pd.DataFrame(
        {
            "feature": contribution_series.index,
            "shap_value": contribution_series.values,
        }
    )

    explanation["impact"] = explanation[
        "shap_value"
    ].apply(
        lambda value:
        "Increases fraud risk"
        if value > 0
        else "Decreases fraud risk"
    )

    return explanation


# -------------------------------------------------------------------
# Main test
# -------------------------------------------------------------------

def main():
    """Run a test explanation."""

    print("=" * 60)
    print("FraudLens AI - Explainable AI")
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

    # Explain one transaction.
    transaction = dataframe.head(
        1
    ).copy()

    print(
        "\nGenerating SHAP explanation..."
    )

    explanation = explain_single_transaction(
        transaction,
        transaction_index=0,
    )

    print(
        "\nTop contributing features:"
    )

    print(
        explanation.head(10).to_string(
            index=False
        )
    )

    print(
        "\nExplainability test completed."
    )


if __name__ == "__main__":
    main()