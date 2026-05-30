# Demo Script

## 30-Second Pitch

This is an explainable fraud alert prioritization system. It scores transactions, ranks alerts by risk, and explains why a transaction should be reviewed. The goal is to simulate a real fraud analyst workflow, not just build a basic prediction model.

## 2-Minute Walkthrough

1. Start with the problem.

```text
Banks process thousands of transactions, and only a small percentage are fraudulent. Fraud teams need to prioritize which alerts to investigate first.
```

2. Show the data workflow.

```text
I started with customer, merchant, and transaction data. I checked missing values, duplicate rows, data types, and fraud distribution before modelling.
```

3. Show feature engineering.

```text
I created fraud-specific features such as velocity risk, geo anomaly, combined risk score, night-device risk, and amount velocity ratio.
```

4. Show model training.

```text
I selected the top 10 features using mutual information, compared Logistic Regression, Random Forest, and Gradient Boosting, then tuned the review threshold for alert prioritization.
```

5. Show the dashboard.

```text
The dashboard works like a fraud analyst workspace. It has command center KPIs, a prioritized alert queue, an investigation view, scenario testing, and model governance.
```

6. Show explainability.

```text
For each transaction, the system explains the main risk drivers, probability impact, severity, and recommended investigation action.
```

## Strong Closing

```text
The main improvement is that the project is now end-to-end: it covers data cleaning, feature engineering, modelling, explainability, alert prioritization, and analyst workflow design.
```

## Questions To Be Ready For

### Why did you use mutual information?

Mutual information helps identify which features carry the most signal about the fraud label. It also keeps the final model explainable because only the top features are used.

### Why is the ROC-AUC around 0.67?

The dataset is synthetic and fraud is naturally imbalanced. The project focuses on explainable prioritization, threshold tuning, and analyst workflow. In production, I would improve performance using more historical customer behavior, merchant history, real-time signals, and larger training data.

### Why tune the threshold?

A fraud system should not blindly use 0.50 as the decision threshold. The threshold controls how many alerts enter manual review and affects the trade-off between catching fraud and overwhelming analysts.

### What makes this project job-ready?

It shows more than modelling. It demonstrates data cleaning, feature engineering, model evaluation, explainability, dashboard engineering, deployment readiness, and business communication.
