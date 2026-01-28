# Customer Churn Prediction — Explainable & Responsible ML Extension

This repository extends an existing customer churn modelling notebook with an **explainability and responsible ML layer** suitable for real-world decision support (risk/retention). The goal is not only high predictive performance, but also **transparent, auditable rationale** for model outputs.

## What this repo demonstrates
- End-to-end churn modelling with **gradient-boosted decision trees**
- Robust evaluation using **ROC-AUC**
- **Class imbalance handling** (SMOTE) with discussion of trade-offs
- **SHAP-based explainability**
  - Global drivers of churn (feature importance + distribution)
  - Local explanations for individual predictions

## Why explainability matters here
Churn predictions are often used to trigger interventions (e.g., retention offers, prioritised outreach). In such settings, we need:
- **accountability** (why did the model flag this customer?)
- **stakeholder trust** (clear drivers, not black-box outputs)
- **risk controls** (bias, proxies, spurious correlations)

## Project structure
- `notebooks/Customer churn prediction.ipynb` — main analysis + explainability
- `requirements.txt` — pinned environment snapshot for reproducibility

## How to run locally
```bash
conda create -n churn-xai python=3.11 -y
conda activate churn-xai
pip install -r requirements.txt
jupyter notebook

Results (summary)

Champion model: (fill from notebook output)

Metric: ROC-AUC on held-out test set

Explainability: SHAP summary plots included in the notebook

Notes on limitations & responsible use

Oversampling (SMOTE) can improve discrimination but may affect probability calibration.

SHAP explanations are model-conditional and depend on the feature representation.

Any deployment should include monitoring, drift checks, and periodic review.

License

MIT (recommended) — or replace with your preferred license.
