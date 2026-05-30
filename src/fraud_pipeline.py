"""Reusable fraud modelling utilities for the dashboard and notebooks.

The project intentionally keeps the full modelling pipeline in one module so a
reviewer can trace the workflow from raw transactions to model explanation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# These are the raw fields available to analysts before a fraud decision exists.
BASE_FEATURES = [
    "channel",
    "transaction_amount_usd",
    "txn_country",
    "txn_hour",
    "device_risk_score",
    "new_device_flag",
    "velocity_1h",
    "velocity_24h",
    "geo_distance_km",
    "merchant_risk_score",
    "is_night_flag",
    "alert_generated",
]

TARGET = "fraud_label"
REVIEW_THRESHOLD = 0.35


@dataclass(frozen=True)
class ModelBundle:
    """Container returned after training the model pipeline."""

    model: Pipeline
    top_features: List[str]
    feature_importance: pd.DataFrame
    metrics: Dict[str, object]
    train_rows: int
    test_rows: int


def resolve_data_path(repo_root: Path) -> Path:
    """Return the transaction file path for either clean or nested repo layouts."""

    candidates = [
        repo_root / "data" / "raw" / "transactions.csv",
        repo_root / "data" / "raw" / "raw" / "transactions.csv",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError("transactions.csv was not found under data/raw.")


def load_transactions(repo_root: Path) -> pd.DataFrame:
    """Load transactions and apply lightweight type validation."""

    transactions = pd.read_csv(resolve_data_path(repo_root), parse_dates=["event_ts"])

    # Normalise expected binary fields in case CSV readers infer them differently.
    binary_cols = ["new_device_flag", "is_night_flag", "alert_generated", TARGET]
    for col in binary_cols:
        transactions[col] = transactions[col].astype(int)

    return transactions


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create analyst-friendly risk features from raw transaction fields."""

    model_df = df.copy()

    # Velocity features are scaled to keep the engineered risk score interpretable.
    max_velocity = max(float(model_df["velocity_24h"].max()), 1.0)
    velocity_component = model_df["velocity_24h"] / max_velocity

    model_df["combined_risk_score"] = (
        model_df["device_risk_score"] * 0.40
        + model_df["merchant_risk_score"] * 0.30
        + velocity_component * 0.20
        + model_df["new_device_flag"] * 0.10
    )

    model_df["high_velocity_flag"] = (model_df["velocity_24h"] >= 10).astype(int)
    model_df["geo_anomaly_flag"] = (model_df["geo_distance_km"] >= 1000).astype(int)
    model_df["night_device_risk"] = (
        model_df["is_night_flag"] * model_df["device_risk_score"]
    )
    model_df["amount_velocity_ratio"] = model_df["transaction_amount_usd"] / (
        model_df["velocity_24h"] + 1
    )

    return model_df


def build_feature_list() -> List[str]:
    """Return the complete feature set used before top-10 selection."""

    return BASE_FEATURES + [
        "combined_risk_score",
        "high_velocity_flag",
        "geo_anomaly_flag",
        "night_device_risk",
        "amount_velocity_ratio",
    ]


def build_model(feature_names: Iterable[str]) -> Pipeline:
    """Create a robust tree model with one-hot encoding for categorical inputs."""

    feature_names = list(feature_names)
    categorical_features = [
        feature for feature in ["channel", "txn_country"] if feature in feature_names
    ]
    numeric_features = [
        feature for feature in feature_names if feature not in categorical_features
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_features,
            ),
            ("numeric", "passthrough", numeric_features),
        ]
    )

    classifier = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def train_top_feature_model(df: pd.DataFrame, top_n: int = 10) -> ModelBundle:
    """Select top mutual-information features, then train on those drivers."""

    model_df = engineer_features(df)
    all_features = build_feature_list()

    X = model_df[all_features]
    y = model_df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    # Mutual information measures how much each feature reduces uncertainty about
    # the fraud label. It is a clear, reviewer-friendly top-feature selection step.
    importances = rank_features_with_mutual_info(X_train, y_train)
    top_features = importances.head(top_n)["feature"].tolist()

    # Second pass: retrain using only the top features requested by the reviewer.
    top_model = build_model(top_features)
    top_model.fit(X_train[top_features], y_train)

    y_pred = top_model.predict(X_test[top_features])
    y_proba = top_model.predict_proba(X_test[top_features])[:, 1]

    metrics = {
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "classification_report": classification_report(
            y_test,
            y_pred,
            output_dict=True,
            zero_division=0,
        ),
    }

    return ModelBundle(
        model=top_model,
        top_features=top_features,
        feature_importance=importances,
        metrics=metrics,
        train_rows=len(X_train),
        test_rows=len(X_test),
    )


def rank_features_with_mutual_info(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Rank original business features with mutual information."""

    encoded = pd.get_dummies(X, columns=["channel", "txn_country"], drop_first=False)
    scores = mutual_info_classif(encoded, y, random_state=42, discrete_features="auto")

    rows = []
    for encoded_feature, score in zip(encoded.columns, scores):
        if encoded_feature.startswith("channel_"):
            feature = "channel"
        elif encoded_feature.startswith("txn_country_"):
            feature = "txn_country"
        else:
            feature = encoded_feature

        rows.append({"feature": feature, "importance": float(score)})

    return (
        pd.DataFrame(rows)
        .groupby("feature", as_index=False)["importance"]
        .sum()
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def aggregate_feature_importance(model: Pipeline, input_features: List[str]) -> pd.DataFrame:
    """Map encoded model importances back to original business feature names."""

    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]
    encoded_names = preprocessor.get_feature_names_out(input_features)
    raw_importance = classifier.feature_importances_

    rows = []
    for encoded_name, importance in zip(encoded_names, raw_importance):
        clean_name = encoded_name.split("__", 1)[-1]

        if clean_name.startswith("channel_"):
            feature = "channel"
        elif clean_name.startswith("txn_country_"):
            feature = "txn_country"
        else:
            feature = clean_name

        rows.append({"feature": feature, "importance": float(importance)})

    importance_df = (
        pd.DataFrame(rows)
        .groupby("feature", as_index=False)["importance"]
        .sum()
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    return importance_df


def predict_transaction(
    bundle: ModelBundle,
    transaction: Dict[str, object],
    reference_df: pd.DataFrame,
) -> Tuple[float, str, pd.DataFrame]:
    """Predict fraud probability and return plain-English feature drivers."""

    input_df = pd.DataFrame([transaction])
    input_df = engineer_features(input_df)
    top_input = input_df[bundle.top_features]

    fraud_probability = float(bundle.model.predict_proba(top_input)[0, 1])
    label = "Fraudulent" if fraud_probability >= REVIEW_THRESHOLD else "Legitimate"
    drivers = explain_with_reference_deltas(top_input.iloc[0], bundle, reference_df)

    return fraud_probability, label, drivers


def explain_with_reference_deltas(
    row: pd.Series,
    bundle: ModelBundle,
    reference_df: pd.DataFrame,
) -> pd.DataFrame:
    """Explain a prediction using top feature importance and population deltas.

    SHAP is used in the notebook for global feature ranking. In the app, this
    fast explanation converts the selected top features into clear analyst notes.
    """

    engineered_reference = engineer_features(reference_df)
    importance_lookup = bundle.feature_importance.set_index("feature")["importance"]
    rows = []

    for feature in bundle.top_features:
        value = row[feature]
        importance = float(importance_lookup.get(feature, 0.0))

        if pd.api.types.is_numeric_dtype(engineered_reference[feature]):
            median = float(engineered_reference[feature].median())
            delta = float(value) - median
            direction = "above" if delta >= 0 else "below"
            reason = f"{feature} is {direction} the dataset median ({median:.3f})."
        else:
            rate = engineered_reference.groupby(feature)[TARGET].mean().get(value, np.nan)
            if pd.isna(rate):
                reason = f"{feature} value '{value}' was not common in the training data."
            else:
                reason = f"{feature}='{value}' has a historical fraud rate of {rate:.1%}."

        rows.append(
            {
                "feature": feature,
                "input_value": value,
                "model_importance": importance,
                "analyst_reason": reason,
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values("model_importance", ascending=False)
        .reset_index(drop=True)
    )
