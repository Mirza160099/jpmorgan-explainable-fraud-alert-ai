"""Feature engineering entry point for the fraud project."""

from __future__ import annotations

from pathlib import Path

from fraud_pipeline import engineer_features, load_transactions


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    transactions = load_transactions(project_root)
    engineered = engineer_features(transactions)
    print(engineered.head())
