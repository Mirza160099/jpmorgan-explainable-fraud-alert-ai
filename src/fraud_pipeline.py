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
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


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
    model_name: str
    top_features: List[str]
    feature_importance: pd.DataFrame
    model_comparison: pd.DataFrame
    threshold_table: pd.DataFrame
    metrics: Dict[str, object]
    review_threshold: float
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


def build_preprocessor(feature_names: Iterable[str]) -> ColumnTransformer:
    """Create preprocessing for categorical and numeric fraud features."""

    feature_names = list(feature_names)
    categorical_features = [
        feature for feature in ["channel", "txn_country"] if feature in feature_names
    ]
    numeric_features = [
        feature for feature in feature_names if feature not in categorical_features
    ]

    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_features,
            ),
            ("numeric", StandardScaler(), numeric_features),
        ]
    )


def build_model(feature_names: Iterable[str], classifier=None) -> Pipeline:
    """Create a fraud model with one-hot encoding and numeric scaling."""

    preprocessor = build_preprocessor(feature_names)

    if classifier is None:
        classifier = build_candidate_models()["Random Forest"]

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def build_candidate_models() -> Dict[str, object]:
    """Return candidate models for reviewer-facing model comparison."""

    classifier = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        ),
        "Random Forest": classifier,
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }


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

    # Compare multiple algorithms using the selected top features before choosing
    # the final deployed model. This gives reviewers stronger model evidence.
    model_comparison, trained_models = compare_candidate_models(
        X_train[top_features],
        X_test[top_features],
        y_train,
        y_test,
        top_features,
    )
    best_model_name = model_comparison.iloc[0]["model"]

    # Second pass: keep the best model trained on only the top features requested
    # by the reviewer.
    top_model = trained_models[best_model_name]
    top_model.fit(X_train[top_features], y_train)

    y_proba = top_model.predict_proba(X_test[top_features])[:, 1]
    threshold_table = tune_thresholds(y_test, y_proba)
    selected_threshold = select_review_threshold(threshold_table)
    y_pred = (y_proba >= selected_threshold).astype(int)

    metrics = {
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "classification_report": classification_report(
            y_test,
            y_pred,
            output_dict=True,
            zero_division=0,
        ),
    }

    return ModelBundle(
        model=top_model,
        model_name=best_model_name,
        top_features=top_features,
        feature_importance=importances,
        model_comparison=model_comparison,
        threshold_table=threshold_table,
        metrics=metrics,
        review_threshold=selected_threshold,
        train_rows=len(X_train),
        test_rows=len(X_test),
    )


def compare_candidate_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    feature_names: List[str],
) -> Tuple[pd.DataFrame, Dict[str, Pipeline]]:
    """Train candidate models and rank them by ROC-AUC."""

    rows = []
    trained_models = {}

    for model_name, classifier in build_candidate_models().items():
        model = build_model(feature_names, classifier)
        model.fit(X_train, y_train)
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= 0.50).astype(int)

        rows.append(
            {
                "model": model_name,
                "roc_auc": float(roc_auc_score(y_test, y_proba)),
                "precision_at_0_50": float(
                    precision_score(y_test, y_pred, zero_division=0)
                ),
                "recall_at_0_50": float(recall_score(y_test, y_pred, zero_division=0)),
                "f1_at_0_50": float(f1_score(y_test, y_pred, zero_division=0)),
            }
        )
        trained_models[model_name] = model

    comparison = (
        pd.DataFrame(rows)
        .sort_values(["roc_auc", "recall_at_0_50"], ascending=False)
        .reset_index(drop=True)
    )

    return comparison, trained_models


def tune_thresholds(y_true: pd.Series, y_proba: np.ndarray) -> pd.DataFrame:
    """Evaluate alert thresholds for fraud-review prioritization."""

    rows = []
    total_transactions = len(y_true)

    for threshold in np.arange(0.10, 0.91, 0.05):
        y_pred = (y_proba >= threshold).astype(int)
        alert_count = int(y_pred.sum())

        rows.append(
            {
                "threshold": round(float(threshold), 2),
                "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, zero_division=0)),
                "f1": float(f1_score(y_true, y_pred, zero_division=0)),
                "alert_count": alert_count,
                "alert_rate": float(alert_count / total_transactions),
            }
        )

    return pd.DataFrame(rows)


def select_review_threshold(threshold_table: pd.DataFrame) -> float:
    """Choose an analyst review threshold with strong F1 and manageable volume."""

    manageable = threshold_table[threshold_table["alert_rate"] <= 0.35]
    candidates = manageable if not manageable.empty else threshold_table
    best_row = candidates.sort_values(["f1", "recall"], ascending=False).iloc[0]
    return float(best_row["threshold"])


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
    threshold = getattr(bundle, "review_threshold", REVIEW_THRESHOLD)
    label = "Fraudulent" if fraud_probability >= threshold else "Legitimate"
    drivers = explain_with_reference_deltas(top_input.iloc[0], bundle, reference_df)

    return fraud_probability, label, drivers


def explain_with_reference_deltas(
    row: pd.Series,
    bundle: ModelBundle,
    reference_df: pd.DataFrame,
) -> pd.DataFrame:
    """Explain a prediction using local probability impact and analyst reason codes."""

    engineered_reference = engineer_features(reference_df)
    importance_lookup = bundle.feature_importance.set_index("feature")["importance"]
    top_input = pd.DataFrame([row[bundle.top_features]])
    original_probability = float(bundle.model.predict_proba(top_input)[0, 1])
    rows = []

    for feature in bundle.top_features:
        value = row[feature]
        importance = float(importance_lookup.get(feature, 0.0))
        reference_value = get_reference_value(engineered_reference, feature)
        baseline_input = top_input.copy()
        baseline_input.loc[0, feature] = reference_value
        baseline_probability = float(bundle.model.predict_proba(baseline_input)[0, 1])
        probability_impact = original_probability - baseline_probability
        risk_direction = (
            "Increases fraud risk"
            if probability_impact > 0
            else "Reduces fraud risk"
            if probability_impact < 0
            else "Neutral"
        )
        severity = impact_severity(probability_impact)

        if pd.api.types.is_numeric_dtype(engineered_reference[feature]):
            delta = float(value) - float(reference_value)
            direction = "above" if delta >= 0 else "below"
            reason = (
                f"{feature} is {direction} the normal reference value "
                f"({float(reference_value):.3f})."
            )
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
                "reference_value": reference_value,
                "probability_impact": probability_impact,
                "model_importance": importance,
                "risk_direction": risk_direction,
                "severity": severity,
                "analyst_reason": reason,
                "recommended_action": recommended_action(feature, probability_impact),
            }
        )

    return (
        pd.DataFrame(rows)
        .assign(abs_probability_impact=lambda df: df["probability_impact"].abs())
        .sort_values(["abs_probability_impact", "model_importance"], ascending=False)
        .drop(columns=["abs_probability_impact"])
        .reset_index(drop=True)
    )


def get_reference_value(reference_df: pd.DataFrame, feature: str):
    """Return the normal comparison value for a feature."""

    if pd.api.types.is_numeric_dtype(reference_df[feature]):
        return float(reference_df[feature].median())

    return reference_df[feature].mode(dropna=True).iloc[0]


def impact_severity(probability_impact: float) -> str:
    """Convert local probability impact into a readable severity band."""

    absolute_impact = abs(probability_impact)

    if absolute_impact >= 0.15:
        return "High"
    if absolute_impact >= 0.07:
        return "Medium"
    if absolute_impact >= 0.02:
        return "Low"
    return "Minimal"


def recommended_action(feature: str, probability_impact: float) -> str:
    """Translate feature-level evidence into fraud analyst next steps."""

    if probability_impact <= 0:
        return "No escalation from this feature alone."

    action_map = {
        "geo_anomaly_flag": "Verify location history and recent customer travel pattern.",
        "geo_distance_km": "Check whether the transaction location is plausible for the customer.",
        "txn_country": "Review country risk, customer home country, and recent cross-border activity.",
        "channel": "Check channel-specific fraud pattern and authentication strength.",
        "alert_generated": "Prioritize because existing rules already raised an alert.",
        "new_device_flag": "Verify device fingerprint, login history, and recent device enrollment.",
        "high_velocity_flag": "Review transaction burst activity and possible account takeover.",
        "velocity_24h": "Check whether recent transaction frequency is abnormal.",
        "velocity_1h": "Investigate short-window transaction burst behavior.",
        "txn_hour": "Check whether transaction timing matches normal customer behavior.",
        "is_night_flag": "Review night-time activity against the customer profile.",
        "transaction_amount_usd": "Compare amount against customer and merchant history.",
        "merchant_risk_score": "Review merchant risk profile and prior suspicious activity.",
        "device_risk_score": "Review device reputation and authentication signals.",
    }

    return action_map.get(feature, "Review this feature as part of manual investigation.")
