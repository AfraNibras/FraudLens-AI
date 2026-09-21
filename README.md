# 🛡️ FraudLens AI

### Explainable AI-Based Fraud Detection and Investigation Platform

FraudLens AI is a machine-learning-powered fraud detection and investigation platform designed to identify suspicious financial transactions, assign transaction risk scores, explain model decisions, and analyze relationships between suspicious accounts.

The platform combines **XGBoost, Logistic Regression, SHAP, NetworkX, Plotly, and Streamlit** into an integrated fraud investigation workflow.

---

## 🚀 Project Overview

Traditional fraud detection systems often focus primarily on predicting whether a transaction is fraudulent.

FraudLens AI extends this approach by combining:

- 🤖 Machine Learning-based fraud detection
- 📊 Transaction risk scoring
- 🧠 Explainable AI using SHAP
- 🔍 Individual transaction investigation
- 🕸️ Fraud relationship and network analysis
- 📈 Interactive analytics
- 🖥️ Streamlit-based investigation dashboard
- 🧪 Automated model and artifact testing

The objective is not only to identify potentially fraudulent transactions, but also to provide supporting evidence that can help an analyst investigate suspicious activity.

---

## ✨ Key Features

### 🤖 Machine Learning Fraud Detection

Two classification models are implemented:

- **Logistic Regression** — baseline model
- **XGBoost** — primary fraud detection model

The XGBoost model generates a fraud probability for each transaction.

---

### 🎯 Risk Scoring

The predicted fraud probability is converted into a 0–100 risk score.

| Risk Score | Category |
|---:|---|
| 0–29.99 | 🟢 LOW |
| 30–59.99 | 🟡 MEDIUM |
| 60–79.99 | 🟠 HIGH |
| 80–100 | 🔴 CRITICAL |

The system also generates an investigation priority and recommended action.

---

### 🧠 Explainable AI

FraudLens AI uses **SHAP (SHapley Additive exPlanations)** to explain individual XGBoost predictions.

The explanation layer identifies:

- Features increasing fraud risk
- Features decreasing fraud risk
- Relative contribution of important features
- Evidence supporting an individual risk prediction

This makes the model's output more interpretable for investigation.

---

### 🕸️ Fraud Network Analysis

The platform uses **NetworkX** to represent suspicious transactions as a directed graph.

- Accounts are represented as nodes.
- Money transfers are represented as edges.
- Transaction count is tracked.
- Total transferred amount is tracked.
- Maximum risk score is tracked.

This helps reveal relationships between accounts involved in suspicious transactions.

---

### 🔍 Transaction Investigation

An analyst can select an individual transaction and inspect:

- Transaction amount
- Transaction type
- Fraud probability
- Risk score
- Risk category
- Investigation priority
- Recommended action

---

### 📊 Interactive Analytics

The Streamlit dashboard provides interactive visualizations for:

- Risk distribution
- Transaction type distribution
- Transaction amount distribution
- Risk score distribution
- Risk by transaction type
- Highest-risk transactions
- Dataset inspection

---

# 🏗️ System Architecture

The complete architecture is:

```text
PaySim Transaction Data
          │
          ▼
Data Preprocessing
          │
          ▼
Feature Engineering
          │
          ▼
Machine Learning
     ┌────┴────┐
     ▼         ▼
Logistic    XGBoost
Regression  Primary Model
     │         │
     └────┬────┘
          ▼
    Model Evaluation
          │
          ▼
   Fraud Prediction
          │
          ▼
      Risk Score
          │
     ┌────┴─────┐
     ▼          ▼
   SHAP      NetworkX
     │          │
     └────┬─────┘
          ▼
 Fraud Investigation
          │
          ▼
 Streamlit Dashboard