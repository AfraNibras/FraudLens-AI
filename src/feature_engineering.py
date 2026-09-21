"""
FraudLens AI - Feature Engineering Module

Creates machine-learning features from the cleaned PaySim dataset.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# -------------------------------------------------------------------
# Project paths
# -------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "paysim_cleaned.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "paysim_features.csv"
)


# -------------------------------------------------------------------
# Load processed dataset
# -------------------------------------------------------------------

def load_processed_data() -> pd.DataFrame:
    """Load the cleaned PaySim dataset."""

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"\nProcessed dataset not found:\n{INPUT_PATH}\n\n"
            "Please run data_preprocessing.py first."
        )

    print(f"Loading processed dataset from:\n{INPUT_PATH}")

    dataframe = pd.read_csv(INPUT_PATH)

    print("\nProcessed dataset loaded successfully.")
    print(f"Rows    : {len(dataframe):,}")
    print(f"Columns : {len(dataframe.columns)}")

    return dataframe


# -------------------------------------------------------------------
# Create transaction features
# -------------------------------------------------------------------

def create_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Create additional features useful for fraud detection.
    """

    dataframe = dataframe.copy()

    print("\nCreating features...")

    # ---------------------------------------------------------------
    # 1. Transaction type encoding
    # ---------------------------------------------------------------

    dataframe["type_encoded"] = (
        dataframe["type"].astype("category").cat.codes
    )

    # ---------------------------------------------------------------
    # 2. Balance changes
    # ---------------------------------------------------------------

    dataframe["origin_balance_change"] = (
        dataframe["oldbalanceOrg"]
        - dataframe["newbalanceOrig"]
    )

    dataframe["destination_balance_change"] = (
        dataframe["newbalanceDest"]
        - dataframe["oldbalanceDest"]
    )

    # ---------------------------------------------------------------
    # 3. Balance discrepancy
    # ---------------------------------------------------------------

    dataframe["origin_balance_error"] = (
        dataframe["oldbalanceOrg"]
        - dataframe["amount"]
        - dataframe["newbalanceOrig"]
    )

    dataframe["destination_balance_error"] = (
        dataframe["oldbalanceDest"]
        + dataframe["amount"]
        - dataframe["newbalanceDest"]
    )

    # ---------------------------------------------------------------
    # 4. Transaction amount relative to origin balance
    # ---------------------------------------------------------------

    dataframe["amount_to_origin_balance"] = (
        dataframe["amount"]
        / (dataframe["oldbalanceOrg"] + 1)
    )

    # ---------------------------------------------------------------
    # 5. Transaction amount relative to destination balance
    # ---------------------------------------------------------------

    dataframe["amount_to_destination_balance"] = (
        dataframe["amount"]
        / (dataframe["oldbalanceDest"] + 1)
    )

    # ---------------------------------------------------------------
    # 6. Zero-balance indicators
    # ---------------------------------------------------------------

    dataframe["origin_zero_balance"] = (
        (dataframe["oldbalanceOrg"] == 0).astype(int)
    )

    dataframe["destination_zero_balance"] = (
        (dataframe["oldbalanceDest"] == 0).astype(int)
    )

    # ---------------------------------------------------------------
    # 7. Large transaction indicator
    # ---------------------------------------------------------------

    amount_threshold = dataframe["amount"].quantile(0.95)

    dataframe["large_transaction"] = (
        dataframe["amount"] >= amount_threshold
    ).astype(int)

    # ---------------------------------------------------------------
    # 8. Log-transformed transaction amount
    # ---------------------------------------------------------------

    dataframe["log_amount"] = np.log1p(
        dataframe["amount"]
    )

    # ---------------------------------------------------------------
    # 9. Time-based features
    # ---------------------------------------------------------------

    dataframe["hour"] = dataframe["step"] % 24

    dataframe["day"] = (
        dataframe["step"] // 24
    )

    dataframe["is_night"] = (
        (dataframe["hour"] < 6)
        | (dataframe["hour"] >= 22)
    ).astype(int)

    # ---------------------------------------------------------------
    # 10. Account relationship indicator
    # ---------------------------------------------------------------

    dataframe["same_origin_destination"] = (
        dataframe["nameOrig"]
        == dataframe["nameDest"]
    ).astype(int)

    print(
        f"\nCreated {len(dataframe.columns)} total columns."
    )

    return dataframe


# -------------------------------------------------------------------
# Validate engineered features
# -------------------------------------------------------------------

def validate_features(dataframe: pd.DataFrame) -> None:
    """Validate the engineered dataset."""

    required_features = [
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

    missing_features = [
        feature
        for feature in required_features
        if feature not in dataframe.columns
    ]

    if missing_features:
        raise ValueError(
            "\nMissing engineered features:\n"
            + "\n".join(
                f"- {feature}"
                for feature in missing_features
            )
        )

    print("\nFeature validation: PASSED")

    numeric_columns = dataframe.select_dtypes(
        include=["number"]
    ).columns

    infinite_values = np.isinf(
        dataframe[numeric_columns]
    ).sum().sum()

    if infinite_values > 0:
        print(
            f"Warning: {infinite_values:,} infinite values found."
        )

        dataframe.replace(
            [np.inf, -np.inf],
            np.nan,
            inplace=True,
        )

        dataframe.fillna(0, inplace=True)

        print("Infinite values handled.")
    else:
        print("Infinite values detected: 0")


# -------------------------------------------------------------------
# Save engineered dataset
# -------------------------------------------------------------------

def save_features(dataframe: pd.DataFrame) -> None:
    """Save the feature-engineered dataset."""

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        f"\nFeature-engineered dataset saved to:\n{OUTPUT_PATH}"
    )


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main() -> None:
    """Run the feature engineering pipeline."""

    print("=" * 60)
    print("FraudLens AI - Feature Engineering")
    print("=" * 60)

    dataframe = load_processed_data()

    dataframe = create_features(dataframe)

    validate_features(dataframe)

    save_features(dataframe)

    print("\nFeature engineering completed successfully.")


if __name__ == "__main__":
    main()