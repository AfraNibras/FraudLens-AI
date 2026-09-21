"""
FraudLens AI - Risk Scoring Module

Converts fraud probabilities into standardized risk scores,
risk categories, investigation priorities, and recommended actions.
"""

from pathlib import Path

import pandas as pd


# -------------------------------------------------------------------
# Project paths
# -------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FEATURE_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "paysim_features.csv"
)


# -------------------------------------------------------------------
# Risk configuration
# -------------------------------------------------------------------

RISK_THRESHOLDS = {
    "LOW": 30,
    "MEDIUM": 60,
    "HIGH": 80,
}


# -------------------------------------------------------------------
# Risk category
# -------------------------------------------------------------------

def classify_risk(risk_score: float) -> str:
    """
    Convert a 0-100 risk score into a risk category.
    """

    if risk_score >= RISK_THRESHOLDS["HIGH"]:
        return "CRITICAL"

    if risk_score >= RISK_THRESHOLDS["MEDIUM"]:
        return "HIGH"

    if risk_score >= RISK_THRESHOLDS["LOW"]:
        return "MEDIUM"

    return "LOW"


# -------------------------------------------------------------------
# Investigation priority
# -------------------------------------------------------------------

def determine_priority(
    risk_score: float,
    fraud_prediction: int,
) -> str:
    """
    Determine investigation priority.

    Priority is based on both the risk score and the model
    prediction.
    """

    if fraud_prediction == 1 and risk_score >= 80:
        return "URGENT"

    if risk_score >= 60:
        return "HIGH"

    if risk_score >= 30:
        return "NORMAL"

    return "LOW"


# -------------------------------------------------------------------
# Recommended action
# -------------------------------------------------------------------

def determine_action(
    risk_category: str,
) -> str:
    """
    Generate an investigation-oriented recommended action.
    """

    actions = {
        "CRITICAL": (
            "Immediately investigate transaction and review "
            "related account activity."
        ),
        "HIGH": (
            "Prioritize transaction for analyst review and "
            "inspect related transactions."
        ),
        "MEDIUM": (
            "Monitor transaction and review additional "
            "behavioral evidence."
        ),
        "LOW": (
            "No immediate investigation required; continue "
            "standard monitoring."
        ),
    }

    return actions.get(
        risk_category,
        "Review transaction manually."
    )


# -------------------------------------------------------------------
# Generate risk assessment
# -------------------------------------------------------------------

def generate_risk_assessment(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate complete risk assessment information.

    Expected columns:
    - risk_score
    - fraud_prediction
    """

    dataframe = dataframe.copy()

    required_columns = [
        "risk_score",
        "fraud_prediction",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            "\nMissing required columns:\n"
            + "\n".join(
                f"- {column}"
                for column in missing_columns
            )
        )

    dataframe["risk_category"] = (
        dataframe["risk_score"]
        .apply(classify_risk)
    )

    dataframe["investigation_priority"] = (
        dataframe.apply(
            lambda row:
            determine_priority(
                row["risk_score"],
                row["fraud_prediction"],
            ),
            axis=1,
        )
    )

    dataframe["recommended_action"] = (
        dataframe["risk_category"]
        .apply(determine_action)
    )

    return dataframe


# -------------------------------------------------------------------
# Generate summary
# -------------------------------------------------------------------

def generate_risk_summary(
    dataframe: pd.DataFrame,
) -> dict:
    """
    Generate high-level risk statistics.
    """

    total_transactions = len(dataframe)

    high_risk_transactions = int(
        dataframe["risk_category"]
        .isin(
            ["HIGH", "CRITICAL"]
        )
        .sum()
    )

    critical_transactions = int(
        (
            dataframe["risk_category"]
            == "CRITICAL"
        ).sum()
    )

    predicted_fraud = int(
        dataframe["fraud_prediction"]
        .sum()
    )

    average_risk = float(
        dataframe["risk_score"]
        .mean()
    )

    return {
        "total_transactions": total_transactions,
        "high_risk_transactions": high_risk_transactions,
        "critical_transactions": critical_transactions,
        "predicted_fraud": predicted_fraud,
        "average_risk_score": average_risk,
    }


# -------------------------------------------------------------------
# Display summary
# -------------------------------------------------------------------

def print_risk_summary(
    summary: dict,
) -> None:
    """Print a readable risk summary."""

    print("\n" + "=" * 60)
    print("FRAUDLENS AI - RISK SUMMARY")
    print("=" * 60)

    print(
        f"Total transactions       : "
        f"{summary['total_transactions']:,}"
    )

    print(
        f"High-risk transactions   : "
        f"{summary['high_risk_transactions']:,}"
    )

    print(
        f"Critical transactions    : "
        f"{summary['critical_transactions']:,}"
    )

    print(
        f"Predicted fraud cases    : "
        f"{summary['predicted_fraud']:,}"
    )

    print(
        f"Average risk score       : "
        f"{summary['average_risk_score']:.2f}"
    )

    print("=" * 60)


# -------------------------------------------------------------------
# Main test
# -------------------------------------------------------------------

def main():
    """
    Run a test of the risk scoring engine.

    Uses the prediction engine on a sample of transactions.
    """

    print("=" * 60)
    print("FraudLens AI - Risk Scoring Engine")
    print("=" * 60)

    if not FEATURE_DATA_PATH.exists():
        raise FileNotFoundError(
            f"\nFeature dataset not found:\n"
            f"{FEATURE_DATA_PATH}"
        )

    # Import prediction functions.
    from predict import (
        predict_transactions,
    )

    print("\nLoading feature dataset...")

    dataframe = pd.read_csv(
        FEATURE_DATA_PATH
    )

    # Use a sample for testing.
    sample = dataframe.head(
        10_000
    ).copy()

    print(
        f"Testing with {len(sample):,} "
        "transactions..."
    )

    # Generate model predictions.
    predictions = predict_transactions(
        sample
    )

    # Generate risk assessment.
    assessed_data = generate_risk_assessment(
        predictions
    )

    # Generate summary.
    summary = generate_risk_summary(
        assessed_data
    )

    print_risk_summary(
        summary
    )

    # Display highest-risk transactions.
    print(
        "\nTop 10 transactions requiring attention:"
    )

    display_columns = [
        "step",
        "type",
        "amount",
        "nameOrig",
        "nameDest",
        "risk_score",
        "risk_category",
        "investigation_priority",
    ]

    top_transactions = (
        assessed_data[
            display_columns
        ]
        .sort_values(
            "risk_score",
            ascending=False,
        )
        .head(10)
    )

    print(
        top_transactions.to_string(
            index=False
        )
    )

    print(
        "\nRisk scoring test completed."
    )


if __name__ == "__main__":
    main()