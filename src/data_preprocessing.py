"""
FraudLens AI - Data Preprocessing Module

Loads the PaySim transaction dataset, performs basic validation,
handles missing values, prepares the target variable, and saves
a cleaned dataset for subsequent feature engineering and modeling.
"""

from pathlib import Path
import pandas as pd


# -------------------------------------------------------------------
# Project paths
# -------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "PS_20174392719_1491204439457_log.csv"
)

PROCESSED_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "paysim_cleaned.csv"
)


# -------------------------------------------------------------------
# Expected dataset columns
# -------------------------------------------------------------------

EXPECTED_COLUMNS = [
    "step",
    "type",
    "amount",
    "nameOrig",
    "oldbalanceOrg",
    "newbalanceOrig",
    "nameDest",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud",
    "isFlaggedFraud",
]


# -------------------------------------------------------------------
# Load dataset
# -------------------------------------------------------------------

def load_dataset(file_path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Load the PaySim CSV dataset.

    Parameters
    ----------
    file_path : Path
        Location of the raw PaySim CSV file.

    Returns
    -------
    pandas.DataFrame
        Loaded transaction dataset.
    """

    if not file_path.exists():
        raise FileNotFoundError(
            f"\nDataset not found.\n"
            f"Expected location:\n{file_path}\n\n"
            f"Please make sure the PaySim CSV is inside:\n"
            f"data/raw/"
        )

    print(f"Loading dataset from:\n{file_path}")

    dataframe = pd.read_csv(file_path)

    print("\nDataset loaded successfully.")
    print(f"Rows    : {len(dataframe):,}")
    print(f"Columns : {len(dataframe.columns)}")

    return dataframe


# -------------------------------------------------------------------
# Validate dataset
# -------------------------------------------------------------------

def validate_dataset(dataframe: pd.DataFrame) -> None:
    """
    Validate that the dataset contains the columns required by
    FraudLens AI.
    """

    missing_columns = [
        column
        for column in EXPECTED_COLUMNS
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            "\nThe dataset is missing the following required columns:\n"
            + "\n".join(f"- {column}" for column in missing_columns)
        )

    print("\nDataset validation: PASSED")


# -------------------------------------------------------------------
# Clean dataset
# -------------------------------------------------------------------

def clean_dataset(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Perform basic cleaning operations.

    Operations:
    - Remove duplicate transactions
    - Handle missing values
    - Convert target variables to integer type
    - Reset dataframe index
    """

    dataframe = dataframe.copy()

    original_rows = len(dataframe)

    # Remove exact duplicate rows.
    dataframe = dataframe.drop_duplicates()

    duplicates_removed = original_rows - len(dataframe)

    if duplicates_removed > 0:
        print(f"Duplicate rows removed: {duplicates_removed:,}")
    else:
        print("Duplicate rows removed: 0")

    # Check missing values.
    missing_values = int(dataframe.isnull().sum().sum())

    if missing_values > 0:
        print(f"Missing values detected: {missing_values:,}")

        # Numeric columns are filled using the median.
        numeric_columns = dataframe.select_dtypes(
            include=["number"]
        ).columns

        for column in numeric_columns:
            if dataframe[column].isnull().any():
                dataframe[column] = dataframe[column].fillna(
                    dataframe[column].median()
                )

        # Categorical columns are filled using the mode.
        categorical_columns = dataframe.select_dtypes(
            exclude=["number"]
        ).columns

        for column in categorical_columns:
            if dataframe[column].isnull().any():
                mode_values = dataframe[column].mode()

                if not mode_values.empty:
                    dataframe[column] = dataframe[column].fillna(
                        mode_values.iloc[0]
                    )
                else:
                    dataframe[column] = dataframe[column].fillna(
                        "UNKNOWN"
                    )

        print("Missing values handled.")
    else:
        print("Missing values detected: 0")

    # Ensure fraud target is integer.
    dataframe["isFraud"] = dataframe["isFraud"].astype(int)

    # Ensure the flagged-fraud indicator is integer.
    dataframe["isFlaggedFraud"] = dataframe[
        "isFlaggedFraud"
    ].astype(int)

    # Reset index after cleaning.
    dataframe = dataframe.reset_index(drop=True)

    return dataframe


# -------------------------------------------------------------------
# Dataset summary
# -------------------------------------------------------------------

def print_dataset_summary(dataframe: pd.DataFrame) -> None:
    """
    Print useful information about the cleaned dataset.
    """

    total_transactions = len(dataframe)

    fraud_transactions = int(dataframe["isFraud"].sum())

    legitimate_transactions = (
        total_transactions - fraud_transactions
    )

    fraud_percentage = (
        fraud_transactions / total_transactions * 100
        if total_transactions > 0
        else 0
    )

    print("\n" + "=" * 60)
    print("FRAUDLENS AI - DATASET SUMMARY")
    print("=" * 60)

    print(f"Total transactions       : {total_transactions:,}")
    print(f"Legitimate transactions  : {legitimate_transactions:,}")
    print(f"Fraudulent transactions  : {fraud_transactions:,}")
    print(f"Fraud percentage         : {fraud_percentage:.4f}%")

    print("\nTransaction types:")
    print(dataframe["type"].value_counts())

    print("\nDataset columns:")
    for column in dataframe.columns:
        print(f"  - {column}")

    print("=" * 60)


# -------------------------------------------------------------------
# Save processed dataset
# -------------------------------------------------------------------

def save_processed_dataset(
    dataframe: pd.DataFrame,
    output_path: Path = PROCESSED_DATA_PATH,
) -> None:
    """
    Save the cleaned dataset.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_csv(
        output_path,
        index=False,
    )

    print(
        f"\nProcessed dataset saved to:\n{output_path}"
    )


# -------------------------------------------------------------------
# Main execution
# -------------------------------------------------------------------

def main() -> None:
    """
    Run the complete preprocessing pipeline.
    """

    print("=" * 60)
    print("FraudLens AI - Data Preprocessing")
    print("=" * 60)

    dataframe = load_dataset()

    validate_dataset(dataframe)

    dataframe = clean_dataset(dataframe)

    print_dataset_summary(dataframe)

    save_processed_dataset(dataframe)

    print("\nPreprocessing completed successfully.")


if __name__ == "__main__":
    main()