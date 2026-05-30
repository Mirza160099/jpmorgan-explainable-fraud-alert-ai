"""Example prediction explanation for a high-risk transaction."""

from __future__ import annotations

from pathlib import Path

from fraud_pipeline import BASE_FEATURES, load_transactions, predict_transaction, train_top_feature_model


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    transactions = load_transactions(project_root)
    bundle = train_top_feature_model(transactions, top_n=10)

    fraud_examples = transactions[transactions["fraud_label"] == 1]
    example_transaction = (
        fraud_examples.sort_values("device_risk_score", ascending=False)
        .iloc[0][BASE_FEATURES]
        .to_dict()
    )

    probability, label, drivers = predict_transaction(
        bundle,
        example_transaction,
        transactions,
    )

    print("Prediction:", label)
    print(f"Fraud probability: {probability:.2%}")
    print(drivers)
