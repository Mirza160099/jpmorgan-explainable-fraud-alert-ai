"""Train the top-feature fraud model from the command line."""

from __future__ import annotations

from pathlib import Path

from fraud_pipeline import load_transactions, train_top_feature_model


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    transactions = load_transactions(project_root)
    bundle = train_top_feature_model(transactions, top_n=10)

    print("Top selected features:")
    for rank, feature in enumerate(bundle.top_features, start=1):
        print(f"{rank}. {feature}")

    print("\nModel comparison:")
    print(bundle.model_comparison.to_string(index=False))

    print(f"\nSelected model: {bundle.model_name}")
    print(f"Selected review threshold: {bundle.review_threshold:.2f}")
    print(f"ROC-AUC: {bundle.metrics['roc_auc']:.4f}")
    print(f"Precision: {bundle.metrics['precision']:.4f}")
    print(f"Recall: {bundle.metrics['recall']:.4f}")
    print(f"F1-score: {bundle.metrics['f1']:.4f}")

    print("\nTop threshold options:")
    print(
        bundle.threshold_table.sort_values("f1", ascending=False)
        .head(5)
        .to_string(index=False)
    )
