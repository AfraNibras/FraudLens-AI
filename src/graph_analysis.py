"""
FraudLens AI - Fraud Network Analysis Module

Builds a transaction relationship graph using NetworkX
and identifies highly connected accounts and suspicious
relationships.
"""

from pathlib import Path

import networkx as nx
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
# Build transaction graph
# -------------------------------------------------------------------

def build_transaction_graph(
    dataframe: pd.DataFrame,
    risk_threshold: float = 60,
) -> nx.DiGraph:
    """
    Build a directed transaction graph.

    Nodes represent accounts.

    Directed edges represent money transfers:
        origin account -> destination account

    Only transactions at or above the supplied risk threshold
    are included in the investigation graph.
    """

    required_columns = [
        "nameOrig",
        "nameDest",
        "amount",
        "risk_score",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            "\nMissing columns required for graph analysis:\n"
            + "\n".join(
                f"- {column}"
                for column in missing_columns
            )
        )

    suspicious_transactions = dataframe[
        dataframe["risk_score"] >= risk_threshold
    ].copy()

    graph = nx.DiGraph()

    for _, transaction in suspicious_transactions.iterrows():

        origin = transaction["nameOrig"]
        destination = transaction["nameDest"]

        amount = float(
            transaction["amount"]
        )

        risk_score = float(
            transaction["risk_score"]
        )

        # Add account nodes.
        graph.add_node(
            origin,
            node_type="origin",
        )

        graph.add_node(
            destination,
            node_type="destination",
        )

        # Add or update transaction edge.
        if graph.has_edge(
            origin,
            destination,
        ):
            graph[origin][destination][
                "transaction_count"
            ] += 1

            graph[origin][destination][
                "total_amount"
            ] += amount

            graph[origin][destination][
                "max_risk_score"
            ] = max(
                graph[origin][destination][
                    "max_risk_score"
                ],
                risk_score,
            )

        else:
            graph.add_edge(
                origin,
                destination,
                transaction_count=1,
                total_amount=amount,
                max_risk_score=risk_score,
            )

    return graph


# -------------------------------------------------------------------
# Account statistics
# -------------------------------------------------------------------

def calculate_account_statistics(
    graph: nx.DiGraph,
) -> pd.DataFrame:
    """
    Calculate connectivity statistics for accounts.
    """

    records = []

    for node in graph.nodes:

        in_degree = graph.in_degree(
            node
        )

        out_degree = graph.out_degree(
            node
        )

        total_degree = (
            in_degree + out_degree
        )

        records.append(
            {
                "account": node,
                "incoming_connections": in_degree,
                "outgoing_connections": out_degree,
                "total_connections": total_degree,
            }
        )

    statistics = pd.DataFrame(
        records
    )

    if statistics.empty:
        return statistics

    statistics = statistics.sort_values(
        "total_connections",
        ascending=False,
    )

    return statistics.reset_index(
        drop=True
    )


# -------------------------------------------------------------------
# Identify highly connected accounts
# -------------------------------------------------------------------

def identify_suspicious_accounts(
    graph: nx.DiGraph,
    minimum_connections: int = 3,
) -> pd.DataFrame:
    """
    Identify accounts with multiple suspicious relationships.
    """

    statistics = calculate_account_statistics(
        graph
    )

    if statistics.empty:
        return statistics

    suspicious_accounts = statistics[
        statistics["total_connections"]
        >= minimum_connections
    ].copy()

    return suspicious_accounts


# -------------------------------------------------------------------
# Graph summary
# -------------------------------------------------------------------

def generate_graph_summary(
    graph: nx.DiGraph,
) -> dict:
    """Generate high-level graph statistics."""

    return {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "connected_components": nx.number_weakly_connected_components(
            graph
        )
        if graph.number_of_nodes() > 0
        else 0,
        "density": nx.density(graph)
        if graph.number_of_nodes() > 1
        else 0,
    }


# -------------------------------------------------------------------
# Main test
# -------------------------------------------------------------------

def main():
    """Run a test of the fraud network analysis."""

    print("=" * 60)
    print("FraudLens AI - Fraud Network Analysis")
    print("=" * 60)

    if not FEATURE_DATA_PATH.exists():
        raise FileNotFoundError(
            f"\nFeature dataset not found:\n"
            f"{FEATURE_DATA_PATH}"
        )

    # Import prediction engine.
    from predict import (
        predict_transactions,
    )

    # Import risk scoring.
    from risk_scoring import (
        generate_risk_assessment,
    )

    print("\nLoading feature dataset...")

    dataframe = pd.read_csv(
        FEATURE_DATA_PATH
    )

    # Use a manageable sample for the first graph.
    sample = dataframe.head(
        10_000
    ).copy()

    print(
        f"Processing {len(sample):,} "
        "transactions..."
    )

    # Generate model predictions.
    predictions = predict_transactions(
        sample
    )

    # Generate risk information.
    assessed_data = generate_risk_assessment(
        predictions
    )

    print(
        "\nBuilding suspicious transaction graph..."
    )

    graph = build_transaction_graph(
        assessed_data,
        risk_threshold=60,
    )

    summary = generate_graph_summary(
        graph
    )

    print("\n" + "=" * 60)
    print("FRAUD NETWORK SUMMARY")
    print("=" * 60)

    print(
        f"Accounts in graph       : "
        f"{summary['nodes']:,}"
    )

    print(
        f"Suspicious relationships: "
        f"{summary['edges']:,}"
    )

    print(
        f"Connected components     : "
        f"{summary['connected_components']:,}"
    )

    print(
        f"Graph density            : "
        f"{summary['density']:.6f}"
    )

    print("=" * 60)

    suspicious_accounts = (
        identify_suspicious_accounts(
            graph,
            minimum_connections=3,
        )
    )

    print(
        "\nHighly connected suspicious accounts:"
    )

    if suspicious_accounts.empty:
        print(
            "No accounts met the connection threshold."
        )
    else:
        print(
            suspicious_accounts.head(10).to_string(
                index=False
            )
        )

    print(
        "\nFraud network analysis completed."
    )


if __name__ == "__main__":
    main()