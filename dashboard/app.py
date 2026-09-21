"""
FraudLens AI - Streamlit Investigation Dashboard

Interactive dashboard for fraud detection,
risk assessment, explainable AI and fraud network analysis.
"""

from pathlib import Path
import sys

import pandas as pd
import plotly.express as px
import streamlit as st


# -------------------------------------------------------------------
# Project configuration
# -------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "paysim_features.csv"
)


# Allow dashboard to import modules from src.
SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(
        0,
        str(SRC_PATH),
    )


from predict import predict_transactions
from risk_scoring import (
    generate_risk_assessment,
)
from explainability import (
    explain_single_transaction,
)
from graph_analysis import (
    build_transaction_graph,
)


# -------------------------------------------------------------------
# Streamlit configuration
# -------------------------------------------------------------------

st.set_page_config(
    page_title="FraudLens AI",
    page_icon="🛡️",
    layout="wide",
)


# -------------------------------------------------------------------
# Application title
# -------------------------------------------------------------------

st.title("🛡️ FraudLens AI")

st.markdown(
    """
### Explainable AI-Based Fraud Detection & Investigation Platform

FraudLens AI combines machine learning, risk scoring,
Explainable AI and transaction relationship analysis
to support fraud investigation.
"""
)


# -------------------------------------------------------------------
# Load data
# -------------------------------------------------------------------

@st.cache_data
def load_data():

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Feature dataset not found:\n{DATA_PATH}"
        )

    return pd.read_csv(
        DATA_PATH
    )


# -------------------------------------------------------------------
# Generate predictions
# -------------------------------------------------------------------

@st.cache_data
def generate_predictions(
    dataframe,
):

    # Use a manageable sample for the dashboard.
    sample = dataframe.head(
        50_000
    ).copy()

    predictions = predict_transactions(
        sample
    )

    assessed = generate_risk_assessment(
        predictions
    )

    return assessed


# -------------------------------------------------------------------
# Load application data
# -------------------------------------------------------------------

try:

    raw_data = load_data()

    data = generate_predictions(
        raw_data
    )

except Exception as error:

    st.error(
        f"Unable to load FraudLens AI data.\n\n"
        f"Error: {error}"
    )

    st.stop()


# -------------------------------------------------------------------
# Sidebar
# -------------------------------------------------------------------

st.sidebar.title(
    "FraudLens AI"
)

st.sidebar.markdown(
    """
**Investigation Platform**

Use the navigation below to explore
fraud risk, transaction explanations
and suspicious account relationships.
"""
)

st.sidebar.markdown(
    """
    <div style="
        text-align: center;
        padding: 10px 0 20px 0;
    ">
        <div style="font-size: 42px;">🛡️</div>
        <h2 style="margin: 0;">FraudLens AI</h2>
        <p style="font-size: 12px; opacity: 0.7;">
            Fraud Detection & Investigation
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown("### 🧭 Navigation")

page = st.sidebar.radio(
    "",
    [
        "🏠  Dashboard",
        "🔍  Transaction Investigation",
        "🧠  AI Explanation",
        "🕸️  Fraud Network",
        "📊  Analytics",
    ],
)

page = page.split("  ", 1)[1]

st.sidebar.markdown("---")

st.sidebar.markdown(
    """
    <div style="
        text-align: center;
        font-size: 12px;
        opacity: 0.65;
        padding-top: 10px;
    ">
        🟢 System Online<br>
        <br>
        FraudLens AI v1.0
    </div>
    """,
    unsafe_allow_html=True,
)


# -------------------------------------------------------------------
# Dashboard
# -------------------------------------------------------------------

if page == "Dashboard":

    st.header(
        "📊 Fraud Investigation Dashboard"
    )

    total_transactions = len(data)

    high_risk = int(
        data["risk_category"]
        .isin(
            ["HIGH", "CRITICAL"]
        )
        .sum()
    )

    critical = int(
        (
            data["risk_category"]
            == "CRITICAL"
        ).sum()
    )

    predicted_fraud = int(
        data["fraud_prediction"]
        .sum()
    )

    average_risk = (
        data["risk_score"]
        .mean()
    )

    # ---------------------------------------------------------------
    # KPI cards
    # ---------------------------------------------------------------

    col1, col2, col3, col4, col5 = st.columns(
        5
    )

    col1.metric(
        "Transactions",
        f"{total_transactions:,}",
    )

    col2.metric(
        "High Risk",
        f"{high_risk:,}",
    )

    col3.metric(
        "Critical",
        f"{critical:,}",
    )

    col4.metric(
        "Predicted Fraud",
        f"{predicted_fraud:,}",
    )

    col5.metric(
        "Average Risk",
        f"{average_risk:.2f}",
    )

    st.divider()

    # ---------------------------------------------------------------
    # Risk distribution
    # ---------------------------------------------------------------

    col1, col2 = st.columns(
        2
    )

    with col1:

        st.subheader(
            "Risk Distribution"
        )

        risk_counts = (
            data["risk_category"]
            .value_counts()
            .reset_index()
        )

        risk_counts.columns = [
            "Risk Category",
            "Count",
        ]

        figure = px.bar(
            risk_counts,
            x="Risk Category",
            y="Count",
            title="Transactions by Risk Category",
        )

        st.plotly_chart(
            figure,
            use_container_width=True,
        )

    with col2:

        st.subheader(
            "Transaction Types"
        )

        type_counts = (
            data["type"]
            .value_counts()
            .reset_index()
        )

        type_counts.columns = [
            "Transaction Type",
            "Count",
        ]

        figure = px.pie(
            type_counts,
            names="Transaction Type",
            values="Count",
            title="Transaction Type Distribution",
        )

        st.plotly_chart(
            figure,
            use_container_width=True,
        )

    # ---------------------------------------------------------------
    # Highest risk transactions
    # ---------------------------------------------------------------

    st.subheader(
        "🚨 Highest-Risk Transactions"
    )

    display_columns = [
        "step",
        "type",
        "amount",
        "nameOrig",
        "nameDest",
        "fraud_probability",
        "risk_score",
        "risk_category",
        "investigation_priority",
    ]

    top_risk = (
        data[
            display_columns
        ]
        .sort_values(
            "risk_score",
            ascending=False,
        )
        .head(20)
    )

    st.dataframe(
        top_risk,
        use_container_width=True,
        hide_index=True,
    )


# -------------------------------------------------------------------
# Transaction Investigation
# -------------------------------------------------------------------

elif page == "Transaction Investigation":

    st.header(
        "🔎 Transaction Investigation"
    )

    transaction_index = st.number_input(
        "Transaction Index",
        min_value=0,
        max_value=len(data) - 1,
        value=0,
        step=1,
    )

    transaction = data.iloc[
        transaction_index
    ]

    st.subheader(
        "Transaction Details"
    )

    col1, col2, col3, col4 = st.columns(
        4
    )

    col1.metric(
        "Amount",
        f"{transaction['amount']:,.2f}",
    )

    col2.metric(
        "Risk Score",
        f"{transaction['risk_score']:.2f}",
    )

    col3.metric(
        "Fraud Probability",
        f"{transaction['fraud_probability'] * 100:.2f}%",
    )

    col4.metric(
        "Risk Category",
        transaction["risk_category"],
    )

    st.divider()

    details = pd.DataFrame(
        {
            "Field": [
                "Transaction Type",
                "Origin Account",
                "Destination Account",
                "Amount",
                "Fraud Prediction",
                "Risk Score",
                "Risk Category",
                "Investigation Priority",
            ],
            "Value": [
                transaction["type"],
                transaction["nameOrig"],
                transaction["nameDest"],
                f"{transaction['amount']:,.2f}",
                int(
                    transaction[
                        "fraud_prediction"
                    ]
                ),
                f"{transaction['risk_score']:.2f}",
                transaction[
                    "risk_category"
                ],
                transaction[
                    "investigation_priority"
                ],
            ],
        }
    )

    st.dataframe(
        details,
        use_container_width=True,
        hide_index=True,
    )

    st.subheader(
        "Recommended Action"
    )

    st.info(
        transaction[
            "recommended_action"
        ]
    )


# -------------------------------------------------------------------
# AI Explanation
# -------------------------------------------------------------------

elif page == "AI Explanation":

    st.header(
        "🧠 Explainable AI"
    )

    transaction_index = st.number_input(
        "Transaction Index",
        min_value=0,
        max_value=len(data) - 1,
        value=0,
        step=1,
    )

    transaction = data.iloc[
        transaction_index
    ]

    st.write(
        f"Transaction risk score: "
        f"**{transaction['risk_score']:.2f}**"
    )

    # SHAP explanation
    explanation = explain_single_transaction(
        data,
        transaction_index=transaction_index,
    )

    st.subheader(
        "Top Risk Factors"
    )

    top_features = (
        explanation
        .head(10)
        .copy()
    )

    st.dataframe(
        top_features,
        use_container_width=True,
        hide_index=True,
    )

    chart_data = (
        top_features
        .sort_values(
            "shap_value"
        )
    )

    figure = px.bar(
        chart_data,
        x="shap_value",
        y="feature",
        orientation="h",
        title="SHAP Feature Contributions",
    )

    st.plotly_chart(
        figure,
        use_container_width=True,
    )

    st.info(
        """
Positive SHAP values indicate features that
push the model toward the fraud class.

Negative SHAP values indicate features that
push the prediction away from the fraud class.
"""
    )


# -------------------------------------------------------------------
# Fraud Network
# -------------------------------------------------------------------

elif page == "Fraud Network":

    st.header(
        "🕸️ Fraud Network Analysis"
    )

    risk_threshold = st.slider(
        "Minimum Risk Score",
        min_value=0,
        max_value=100,
        value=60,
        step=5,
    )

    suspicious_data = data[
        data["risk_score"]
        >= risk_threshold
    ].copy()

    st.write(
        f"Transactions included in network: "
        f"**{len(suspicious_data):,}**"
    )

    if suspicious_data.empty:

        st.warning(
            "No transactions meet the selected risk threshold."
        )

    else:

        graph = build_transaction_graph(
            suspicious_data,
            risk_threshold=risk_threshold,
        )

        st.metric(
            "Accounts",
            graph.number_of_nodes(),
        )

        st.metric(
            "Relationships",
            graph.number_of_edges(),
        )

        # -----------------------------------------------------------
        # Build edge dataframe for Plotly
        # -----------------------------------------------------------

        edges = []

        for source, target, attributes in graph.edges(
            data=True
        ):

            edges.append(
                {
                    "Origin": source,
                    "Destination": target,
                    "Transactions": attributes[
                        "transaction_count"
                    ],
                    "Total Amount": attributes[
                        "total_amount"
                    ],
                    "Maximum Risk": attributes[
                        "max_risk_score"
                    ],
                }
            )

        if edges:

            edge_dataframe = pd.DataFrame(
                edges
            )

            st.subheader(
                "Suspicious Relationships"
            )

            st.dataframe(
                edge_dataframe.head(100),
                use_container_width=True,
                hide_index=True,
            )

        else:

            st.info(
                "No suspicious relationships found."
            )


# -------------------------------------------------------------------
# Analytics
# -------------------------------------------------------------------

elif page == "Analytics":

    st.header(
        "📈 Fraud Analytics"
    )

    # ---------------------------------------------------------------
    # Amount distribution
    # ---------------------------------------------------------------

    st.subheader(
        "Transaction Amount Distribution"
    )

    figure = px.histogram(
        data,
        x="amount",
        nbins=50,
        title="Transaction Amount Distribution",
    )

    st.plotly_chart(
        figure,
        use_container_width=True,
    )

    # ---------------------------------------------------------------
    # Risk score distribution
    # ---------------------------------------------------------------

    st.subheader(
        "Risk Score Distribution"
    )

    figure = px.histogram(
        data,
        x="risk_score",
        nbins=50,
        title="Risk Score Distribution",
    )

    st.plotly_chart(
        figure,
        use_container_width=True,
    )

    # ---------------------------------------------------------------
    # Transaction type vs risk
    # ---------------------------------------------------------------

    st.subheader(
        "Risk by Transaction Type"
    )

    risk_by_type = (
        data.groupby("type")[
            "risk_score"
        ]
        .mean()
        .reset_index()
        .sort_values(
            "risk_score",
            ascending=False,
        )
    )

    figure = px.bar(
        risk_by_type,
        x="type",
        y="risk_score",
        title="Average Risk Score by Transaction Type",
    )

    st.plotly_chart(
        figure,
        use_container_width=True,
    )

    # ---------------------------------------------------------------
    # Data preview
    # ---------------------------------------------------------------

    st.subheader(
        "Investigation Dataset"
    )

    st.dataframe(
        data.head(100),
        use_container_width=True,
        hide_index=True,
    )