# FraudLens AI — System Architecture

## 1. Overview

FraudLens AI is an explainable AI-based fraud detection and investigation platform designed to identify suspicious financial transactions and provide interpretable evidence for investigation.

The system combines machine learning, explainable AI, behavioral feature engineering, risk scoring, and graph-based relationship analysis through an interactive Streamlit dashboard.

---

## 2. High-Level Architecture

```text
┌──────────────────────────────────────────────────────────────┐
│                     PaySim Transaction Data                  │
│                                                              │
│  step | type | amount | accounts | balances | fraud label    │
└──────────────────────────────┬───────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│                    Data Preprocessing                        │
│                                                              │
│  • Missing-value handling                                    │
│  • Duplicate removal                                         │
│  • Data validation                                           │
│  • Data type conversion                                      │
└──────────────────────────────┬───────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│                    Feature Engineering                       │
│                                                              │
│  • Balance-change features                                   │
│  • Balance-error features                                    │
│  • Amount ratios                                             │
│  • Zero-balance indicators                                   │
│  • Transaction-size indicators                               │
│  • Time-based features                                       │
│  • Encoded transaction type                                  │
└──────────────────────────────┬───────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│                     Machine Learning                         │
│                                                              │
│  Logistic Regression          XGBoost                        │
│  (Baseline Model)             (Primary Model)                │
│             │                         │                      │
│             └──────────────┬──────────┘                      │
│                            ▼                                 │
│                    Model Evaluation                          │
└──────────────────────────────┬───────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│                    Fraud Prediction                          │
│                                                              │
│  • Fraud probability                                         │
│  • Binary fraud prediction                                   │
│  • Transaction risk score                                    │
└──────────────────────────────┬───────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│                     Risk Engine                              │
│                                                              │
│  LOW → MEDIUM → HIGH → CRITICAL                              │
│                                                              │
│  Investigation priority and recommended action               │
└───────────────────────┬───────────────────────┬──────────────┘
                        │                       │
             ┌──────────▼──────────┐   ┌──────▼──────────────┐
             │   Explainable AI    │   │  Graph Analysis     │
             │                     │   │                     │
             │       SHAP          │   │     NetworkX        │
             │                     │   │                     │
             │ • Feature impact    │   │ • Account nodes     │
             │ • Risk contributors │   │ • Relationships     │
             │ • Risk reducers     │   │ • Suspicious links  │
             └──────────┬──────────┘   └──────┬──────────────┘
                        │                       │
                        └───────────┬───────────┘
                                    ▼
┌──────────────────────────────────────────────────────────────┐
│                 FraudLens AI Dashboard                       │
│                    Streamlit + Plotly                        │
│                                                              │
│  Dashboard                                                   │
│  Transaction Investigation                                   │
│  AI Explanation                                              │
│  Fraud Network                                               │
│  Analytics                                                   │
└──────────────────────────────────────────────────────────────┘