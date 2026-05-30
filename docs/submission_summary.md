# Submission Summary

## Short Version

I upgraded the JPMorgan fraud project into an explainable fraud alert prioritization system. The project now includes data cleaning, feature engineering, top-10 feature selection, model comparison, threshold tuning, local explainability, and a Streamlit fraud analyst workspace.

## Full Version

This project simulates how a financial institution could prioritize fraud alerts for analyst review. I started by validating the raw customer, merchant, and transaction datasets, checking missing values, duplicates, data types, and fraud label distribution.

After data validation, I engineered fraud-specific features such as velocity risk, geographic anomaly, night-device risk, combined risk score, and amount-to-velocity behavior. I then used mutual information to select the top 10 most useful fraud drivers before modelling.

For modelling, I compared Logistic Regression, Random Forest, and Gradient Boosting using ROC-AUC, precision, recall, and F1-score. I selected the strongest model and added threshold tuning so the system can balance fraud detection with investigation workload.

The final application is a Streamlit fraud analyst workspace. It includes command center KPIs, a prioritized alert queue, critical/review/monitor tiers, SLA hours, transaction-level investigation, manual scenario testing, model evidence, and local reason-code explanations.

The explainability layer does not only show a fraud probability. It explains which features influenced the prediction, compares values against normal references, estimates probability impact, assigns severity, and recommends an analyst action.

## Key Technical Contributions

- Cleaned and validated raw datasets.
- Documented the data cleaning process step by step.
- Engineered fraud-specific risk features.
- Selected top 10 model features using mutual information.
- Compared multiple machine learning models.
- Tuned the fraud review threshold.
- Built a reusable modelling pipeline in `src/fraud_pipeline.py`.
- Created an analyst-style dashboard in Streamlit.
- Added local explanation reason codes and recommended actions.
- Prepared the project for Railway deployment.

## What To Say In A Demo

```text
This project is an explainable fraud alert prioritization system. The goal is not only to predict fraud, but to help analysts decide which transactions to investigate first and why.

I validated the raw datasets, engineered fraud-specific features, selected the top 10 drivers, compared multiple models, tuned the review threshold, and built a Streamlit analyst workspace.

The dashboard includes command center KPIs, an alert queue, transaction investigation, scenario testing, and model governance. For each prediction, the system gives reason codes, probability impact, severity, and recommended analyst actions.
```

## If Asked About Model Performance

```text
The selected model is a Random Forest trained on the top 10 selected features. ROC-AUC is around 0.67 on the current synthetic dataset. I focused on explainable alert prioritization, threshold tuning, and analyst workflow rather than only maximizing accuracy.
```

## If Asked Why Explainability Matters

```text
In financial services, fraud models need to be reviewable. An analyst should understand why a transaction was prioritized. That is why the project includes local explanations, reference comparisons, probability impact, severity, and recommended actions.
```

## If Asked What You Would Improve Next

```text
The next improvements would be adding real-time streaming alerts, customer-level historical profiles, model monitoring, drift detection, and deployment on a production cloud stack.
```
