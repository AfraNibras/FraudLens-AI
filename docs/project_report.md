# FraudLens AI

## An Explainable AI-Based Fraud Detection and Investigation Platform

---

# 1. Abstract

Financial fraud is a major challenge in digital financial systems because fraudulent transactions can occur rapidly and may involve complex relationships between multiple accounts.

FraudLens AI is an end-to-end machine learning-based fraud detection and investigation platform designed to identify potentially fraudulent transactions and provide interpretable information for investigation.

The system uses the PaySim synthetic financial transaction dataset and applies data preprocessing, behavioral feature engineering, supervised machine learning, fraud probability estimation, risk scoring, Explainable AI, and graph-based relationship analysis.

Two machine learning approaches are implemented: Logistic Regression as a baseline model and XGBoost as the primary nonlinear classification model.

The platform uses SHAP to explain individual model predictions by identifying features that increase or decrease fraud risk. NetworkX is used to analyze suspicious relationships between origin and destination accounts.

A Streamlit-based interactive dashboard integrates the different components into a single investigation interface.

The resulting platform demonstrates how artificial intelligence and explainable machine learning can be combined to support transaction-level fraud investigation.

---

# 2. Introduction

The growth of digital financial transactions has increased the importance of automated fraud detection systems.

Traditional rule-based systems can identify known suspicious patterns, but they may struggle with complex or previously unseen transaction behavior.

Machine learning provides an alternative approach by learning patterns from historical transaction data.

However, a machine learning prediction alone may not be sufficient for an investigation. An analyst may need to understand:

- Why was a transaction considered risky?
- Which transaction characteristics influenced the prediction?
- What is the estimated fraud probability?
- Which accounts are connected to suspicious activity?
- What level of investigation priority should be assigned?

FraudLens AI addresses these requirements by combining fraud detection with explainability, risk scoring, and relationship analysis.

---

# 3. Problem Statement

Financial transaction datasets can contain a very large number of legitimate transactions and a relatively small number of fraudulent transactions.

This creates several challenges:

1. Fraudulent transactions are difficult to identify automatically.
2. Fraud detection datasets are highly imbalanced.
3. A binary fraud prediction provides limited investigation context.
4. Machine learning models can be difficult to interpret.
5. Fraud may involve relationships between multiple accounts.
6. Analysts require tools for transaction-level investigation.

Therefore, there is a need for an integrated system capable of detecting suspicious transactions while also providing risk assessment, explanations, and relationship analysis.

---

# 4. Proposed Solution

FraudLens AI proposes an end-to-end fraud investigation platform.

The system performs the following operations:

```text
Transaction Dataset
        ↓
Data Preprocessing
        ↓
Feature Engineering
        ↓
Machine Learning
        ↓
Fraud Probability
        ↓
Risk Scoring
        ↓
Explainable AI
        ↓
Relationship Analysis
        ↓
Interactive Investigation Dashboard