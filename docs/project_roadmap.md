# Explainable Fraud Alert Prioritization System - Project Roadmap

## Project Goal

Build a professional fraud alert prioritization system inspired by banking fraud investigation workflows. The system should not only predict whether a transaction is likely to be fraudulent, but also explain the decision in a way a fraud analyst, reviewer, or recruiter can understand.

## Problem Statement

Banks process large volumes of digital transactions. A small percentage may be fraudulent, but fraud teams cannot manually investigate every transaction with equal urgency. This project focuses on prioritizing risky transactions, identifying the strongest fraud drivers, and explaining why a transaction should be reviewed.

## Current Baseline

The project currently includes:

- Raw customer, merchant, and transaction datasets.
- Data cleaning checks for missing values, duplicate records, and data types.
- Feature engineering for transaction velocity, geography anomaly, device risk, merchant risk, and night-time behavior.
- Top-10 feature selection using mutual information.
- A machine learning fraud model using the selected top features.
- A Streamlit dashboard for monitoring transactions and predicting fraud risk.
- Analyst-friendly explanation reasons for each prediction.
- Railway deployment configuration.
- A fraud analyst workspace with command center KPIs, a prioritized alert queue, transaction investigation, scenario testing, and model governance.

## Target Big-League Version

The strongest version of this project should include:

1. A complete notebook story from business problem to deployment.
2. A clean, reusable Python pipeline instead of one-off notebook code.
3. Model comparison across multiple algorithms.
4. Threshold tuning for fraud alert prioritization.
5. Explainability using global feature importance and local transaction explanations.
6. A polished analyst dashboard with:
   - executive KPIs
   - alert queue
   - transaction input form
   - fraud probability
   - top reason codes
   - model evidence
7. Clear GitHub documentation with screenshots, architecture, setup, and deployment instructions.
8. A short presentation/submission story explaining the project professionally.

## Step-by-Step Build Plan

### Phase 1 - Foundation

- Organize project folders.
- Validate raw datasets.
- Document the problem statement and approach.
- Confirm missing values, duplicates, and data types.

### Phase 2 - Data Understanding

- Explore fraud vs legitimate transaction distribution.
- Analyze amount, channel, country, velocity, device risk, merchant risk, and geography.
- Identify early fraud patterns.

### Phase 3 - Feature Engineering

- Create fraud-specific model features:
  - combined risk score
  - high velocity flag
  - geo anomaly flag
  - night device risk
  - amount velocity ratio
- Validate whether engineered features behave differently for fraud and legitimate cases.

### Phase 4 - Feature Selection

- Rank features using mutual information.
- Keep the top 10 features for interpretability.
- Explain why feature selection matters for analyst trust.

### Phase 5 - Modelling

- Train a baseline model.
- Compare stronger models.
- Evaluate using ROC-AUC, precision, recall, and F1-score.
- Tune the review threshold for alert prioritization.

### Phase 6 - Explainability

- Show global top features.
- Explain individual predictions with reason codes.
- Translate technical model output into fraud analyst language.

### Phase 7 - Dashboard

- Build a polished Streamlit app with:
  - monitoring view
  - prioritized alert queue
  - transaction investigation view
  - scenario testing form
  - top features
  - explanation table
  - model evidence

Status: complete. The dashboard now uses a fraud analyst workspace layout with command center KPIs, triage controls, alert queue tiers, SLA hours, transaction investigation, scenario testing, and model governance.

### Phase 8 - Deployment And Submission

- Push final code to GitHub.
- Deploy the app.
- Add screenshots.
- Prepare a short explanation for professor/recruiter review.

Status: in progress. GitHub documentation has been upgraded with a polished README, architecture notes, submission summary, and demo script. The next step is to capture fresh screenshots from the final dashboard and push the final project to GitHub.

## Documentation Files

- `README.md`: main GitHub project page.
- `docs/architecture.md`: system architecture and workflow explanation.
- `docs/submission_summary.md`: professor/reviewer submission wording.
- `docs/demo_script.md`: short project presentation script.
- `docs/project_roadmap.md`: build plan and progress tracker.

## Current Professor Update

I have moved the project from a basic dataset/dashboard exercise toward a complete explainable fraud alert prioritization system. The current version includes data cleaning documentation, feature engineering, top-10 feature selection, model training, explainability, and an interactive dashboard where users can input transaction details and receive a fraud prediction with reasons.

## Next Improvement

The next improvement is to strengthen the modelling section by adding model comparison and threshold tuning, then upgrading the dashboard to look more like a professional fraud analyst workspace.
