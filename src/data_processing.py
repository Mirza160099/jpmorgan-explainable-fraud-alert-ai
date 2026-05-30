"""Data loading and validation helpers for the fraud project."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from fraud_pipeline import load_transactions


def data_quality_report(repo_root: Path) -> pd.DataFrame:
    """Return missing-value and duplicate checks for the raw datasets."""

    raw_dir = repo_root / "data" / "raw"
    nested_dir = raw_dir / "raw"
    data_dir = raw_dir if (raw_dir / "customers.csv").exists() else nested_dir

    report_rows = []
    for file_name in ["customers.csv", "merchants.csv", "transactions.csv"]:
        df = pd.read_csv(data_dir / file_name)
        report_rows.append(
            {
                "dataset": file_name,
                "rows": len(df),
                "columns": len(df.columns),
                "missing_values": int(df.isnull().sum().sum()),
                "duplicate_rows": int(df.duplicated().sum()),
            }
        )

    return pd.DataFrame(report_rows)


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    print(data_quality_report(project_root))
    print(load_transactions(project_root).head())
