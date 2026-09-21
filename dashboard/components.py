import streamlit as st
import plotly.express as px


def display_kpi_cards(
    total_transactions,
    high_risk_transactions,
    critical_transactions,
    predicted_fraud,
):
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Transactions",
            f"{total_transactions:,}"
        )

    with col2:
        st.metric(
            "High-Risk Transactions",
            f"{high_risk_transactions:,}"
        )

    with col3:
        st.metric(
            "Critical Transactions",
            f"{critical_transactions:,}"
        )

    with col4:
        st.metric(
            "Predicted Fraud",
            f"{predicted_fraud:,}"
        )


def create_risk_distribution_chart(data):
    if "risk_category" not in data.columns:
        return None

    risk_counts = (
        data["risk_category"]
        .value_counts()
        .reset_index()
    )

    risk_counts.columns = ["Risk Category", "Count"]

    fig = px.bar(
        risk_counts,
        x="Risk Category",
        y="Count",
        title="Risk Distribution",
        text="Count",
    )

    fig.update_layout(
        xaxis_title="Risk Category",
        yaxis_title="Number of Transactions",
    )

    return fig


def create_transaction_type_chart(data):
    if "type" not in data.columns:
        return None

    type_counts = (
        data["type"]
        .value_counts()
        .reset_index()
    )

    type_counts.columns = ["Transaction Type", "Count"]

    fig = px.pie(
        type_counts,
        names="Transaction Type",
        values="Count",
        title="Transaction Type Distribution",
    )

    return fig


def create_risk_score_histogram(data):
    if "risk_score" not in data.columns:
        return None

    fig = px.histogram(
        data,
        x="risk_score",
        nbins=30,
        title="Risk Score Distribution",
    )

    fig.update_layout(
        xaxis_title="Risk Score",
        yaxis_title="Number of Transactions",
    )

    return fig


def create_amount_histogram(data):
    if "amount" not in data.columns:
        return None

    fig = px.histogram(
        data,
        x="amount",
        nbins=50,
        title="Transaction Amount Distribution",
    )

    fig.update_layout(
        xaxis_title="Transaction Amount",
        yaxis_title="Number of Transactions",
    )

    return fig