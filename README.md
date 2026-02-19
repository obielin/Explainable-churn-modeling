# Customer Churn Prediction — Explainable & Responsible ML

This project extends a customer churn modelling workflow with an integrated **explainability and responsible ML layer**.

The objective is not only strong predictive performance, but also **transparent, auditable decision support** suitable for real-world retention use cases.

---

## What This Project Covers

- End-to-end churn classification using **gradient-boosted decision trees**
- Evaluation using **ROC-AUC** on a held-out test set
- **Class imbalance handling (SMOTE)** with documented trade-offs
- Integrated **SHAP explainability**
  - Global drivers of churn
  - Local explanations for individual predictions
- Explicit documentation of responsible ML considerations

---

## Modelling Overview

### Data Preparation
- Deterministic train/test split
- Feature preprocessing and encoding
- SMOTE applied to improve minority class detection

### Model
- Gradient-boosted decision tree classifier
- Suitable for non-linear feature interactions
- Compatible with SHAP TreeExplainer

### Evaluation
- Primary metric: **ROC-AUC**
- Confusion matrix and threshold analysis included
- No test data leakage during training

---

## Explainability (SHAP)

### Global Explanations
- Ranked feature importance
- Directional impact on churn risk
- Distribution of feature effects

### Local Explanations
- Customer-level feature contributions
- Supports case-level auditability
- Enables human-in-the-loop review

This shifts the model from a black-box predictor to a transparent decision-support system.

---

## Responsible ML Notes

- SMOTE may affect probability calibration
- SHAP explanations are model-conditional (not causal)
- Any production deployment should include:
  - Probability calibration checks
  - Threshold optimisation
  - Drift monitoring
  - Fairness evaluation

---

## Run Locally

```bash
conda create -n churn-xai python=3.11 -y
conda activate churn-xai
pip install -r requirements.txt
jupyter notebook
