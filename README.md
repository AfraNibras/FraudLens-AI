# FraudLens AI

## Explainable AI-Based Fraud Detection & Investigation Platform

FraudLens AI is an end-to-end machine learning and explainable AI platform designed to detect potentially fraudulent financial transactions, assign risk levels, investigate individual transactions, explain model decisions, and analyze suspicious relationships between accounts.

The platform combines machine learning, behavioral feature engineering, risk scoring, SHAP-based explainability, graph analysis, and an interactive Streamlit dashboard into a unified fraud investigation workflow.

---

## Project Overview

Traditional fraud detection systems often provide only a binary prediction such as:

> Fraud / Not Fraud

FraudLens AI extends this approach by providing an investigation-oriented workflow.

For each transaction, the platform can provide:

- Fraud probability
- Risk score
- Risk category
- Investigation priority
- Transaction-level details
- Explainable AI factors
- Suspicious account relationships
- Interactive visual analytics

The goal is to support analysts in understanding **why a transaction may be suspicious**, rather than relying only on a machine learning prediction.

---

## Key Features

### 1. Machine Learning Fraud Detection

FraudLens AI uses supervised machine learning models to identify potentially fraudulent transactions.

Implemented models include:

- Logistic Regression
- XGBoost

The system uses class balancing techniques because fraudulent transactions are significantly less common than legitimate transactions in the PaySim dataset.

---

### 2. Behavioral Feature Engineering

The system derives additional behavioral and financial features from the original transaction data.

Examples include:

- Origin balance change
- Destination balance change
- Origin balance error
- Destination balance error
- Amount-to-origin-balance ratio
- Amount-to-destination-balance ratio
- Zero-balance indicators
- Large transaction indicator
- Log-transformed transaction amount
- Transaction hour
- Transaction day
- Night-time indicator
- Same origin/destination indicator

These features provide additional behavioral information to the machine learning model.

---

### 3. Fraud Risk Scoring

Instead of displaying only a binary prediction, FraudLens AI converts the model probability into a risk score from 0 to 100.

Risk categories:

| Risk Score | Category |
|------------|----------|
| 0–29 | LOW |
| 30–59 | MEDIUM |
| 60–79 | HIGH |
| 80–100 | CRITICAL |

The platform also assigns an investigation priority based on the detected risk.

---

### 4. Explainable AI

FraudLens AI uses SHAP (SHapley Additive exPlanations) to explain individual model predictions.

The Explainable AI module identifies:

- Features increasing fraud risk
- Features decreasing fraud risk
- Strength of each feature contribution
- Transaction-specific model reasoning

This makes the machine learning model more interpretable for investigation purposes.

---

### 5. Transaction Investigation

The Transaction Investigation module allows users to inspect individual transactions.

The dashboard provides:

- Transaction amount
- Transaction type
- Origin account
- Destination account
- Fraud prediction
- Fraud probability
- Risk score
- Risk category
- Investigation priority
- Recommended investigation action

---

### 6. Fraud Network Analysis

Fraudulent activity can involve relationships between multiple accounts.

FraudLens AI uses NetworkX to model suspicious transaction relationships as a graph.

The graph represents:

```text
Origin Account → Destination Account