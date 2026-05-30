"""Streamlit fraud analyst dashboard.

Run locally with:
    streamlit run src/dashboard.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from fraud_pipeline import (
    REVIEW_THRESHOLD,
    load_transactions,
    predict_transaction,
    train_top_feature_model,
)


# Streamlit reruns this file on every interaction, so expensive data/model work
# is cached and reused until the source data changes.
REPO_ROOT = Path(__file__).resolve().parents[1]


@st.cache_data(show_spinner=False)
def get_transactions() -> pd.DataFrame:
    """Load the raw transaction dataset once for dashboard use."""

    return load_transactions(REPO_ROOT)


@st.cache_resource(show_spinner="Training top-feature fraud model...")
def get_model_bundle(transactions: pd.DataFrame):
    """Train and cache the top-10-feature model for interactive predictions."""

    return train_top_feature_model(transactions, top_n=10)


def binary_label(value: int) -> str:
    """Display binary indicators as analyst-friendly text."""

    return "Yes" if int(value) == 1 else "No"


def render_prediction_form(transactions: pd.DataFrame, bundle) -> None:
    """Render the user-input transaction form and prediction output."""

    st.subheader("Transaction Fraud Prediction")

    with st.form("transaction_prediction_form"):
        st.caption("Enter a transaction profile. The model uses only the selected top-10 features.")

        col1, col2, col3 = st.columns(3)

        with col1:
            channel = st.selectbox(
                "Channel",
                sorted(transactions["channel"].dropna().unique()),
            )
            amount = st.number_input(
                "Transaction amount (USD)",
                min_value=0.0,
                value=float(transactions["transaction_amount_usd"].median()),
                step=10.0,
            )
            country = st.selectbox(
                "Transaction country",
                sorted(transactions["txn_country"].dropna().unique()),
            )
            txn_hour = st.slider("Transaction hour", 0, 23, 12)

        with col2:
            device_risk = st.slider("Device risk score", 0.0, 1.0, 0.55, 0.01)
            merchant_risk = st.slider("Merchant risk score", 0.0, 1.0, 0.45, 0.01)
            geo_distance = st.number_input(
                "Geo distance (km)",
                min_value=0.0,
                value=float(transactions["geo_distance_km"].median()),
                step=25.0,
            )
            velocity_24h = st.number_input(
                "Transactions in last 24h",
                min_value=0,
                value=int(transactions["velocity_24h"].median()),
                step=1,
            )

        with col3:
            velocity_1h = st.number_input(
                "Transactions in last 1h",
                min_value=0,
                value=int(transactions["velocity_1h"].median()),
                step=1,
            )
            new_device = st.selectbox("New device?", [0, 1], format_func=binary_label)
            is_night = st.selectbox("Night transaction?", [0, 1], format_func=binary_label)
            alert_generated = st.selectbox(
                "Existing alert generated?",
                [0, 1],
                format_func=binary_label,
            )

        submitted = st.form_submit_button("Predict Fraud Risk", type="primary")

    if not submitted:
        return

    # The app collects every base feature, then the pipeline derives engineered
    # fields and keeps only the top selected drivers for the final prediction.
    transaction = {
        "channel": channel,
        "transaction_amount_usd": amount,
        "txn_country": country,
        "txn_hour": txn_hour,
        "device_risk_score": device_risk,
        "new_device_flag": new_device,
        "velocity_1h": velocity_1h,
        "velocity_24h": velocity_24h,
        "geo_distance_km": geo_distance,
        "merchant_risk_score": merchant_risk,
        "is_night_flag": is_night,
        "alert_generated": alert_generated,
    }

    fraud_probability, label, drivers = predict_transaction(
        bundle,
        transaction,
        transactions,
    )

    left, right = st.columns([1, 2])
    left.metric("Prediction", label)
    left.metric("Fraud probability", f"{fraud_probability:.1%}")

    if fraud_probability >= REVIEW_THRESHOLD:
        left.error("Prioritize for analyst investigation.")
    else:
        left.success("No immediate fraud escalation based on current inputs.")

    right.dataframe(
        drivers[["feature", "input_value", "model_importance", "analyst_reason"]],
        use_container_width=True,
        hide_index=True,
    )


def main() -> None:
    """Build the fraud monitoring and prediction experience."""

    st.set_page_config(
        page_title="Explainable Fraud Alert Prioritization",
        layout="wide",
    )

    st.title("Explainable Fraud Alert Prioritization System")
    st.caption(
        "Enterprise-style fraud monitoring, top-feature modelling, and analyst explanations."
    )

    transactions = get_transactions()
    bundle = get_model_bundle(transactions)

    with st.sidebar:
        st.header("Monitoring Filters")
        country_filter = st.multiselect(
            "Transaction country",
            sorted(transactions["txn_country"].dropna().unique()),
            default=sorted(transactions["txn_country"].dropna().unique()),
        )
        channel_filter = st.multiselect(
            "Channel",
            sorted(transactions["channel"].dropna().unique()),
            default=sorted(transactions["channel"].dropna().unique()),
        )
        fraud_only = st.checkbox("Show fraud only")
        min_device_risk = st.slider("Minimum device risk", 0.0, 1.0, 0.0, 0.05)

    filtered = transactions[
        transactions["txn_country"].isin(country_filter)
        & transactions["channel"].isin(channel_filter)
        & (transactions["device_risk_score"] >= min_device_risk)
    ].copy()

    if fraud_only:
        filtered = filtered[filtered["fraud_label"] == 1]

    total_transactions = len(filtered)
    total_fraud = int(filtered["fraud_label"].sum()) if total_transactions else 0
    fraud_rate = (total_fraud / total_transactions * 100) if total_transactions else 0

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Filtered transactions", f"{total_transactions:,}")
    kpi2.metric("Fraudulent transactions", f"{total_fraud:,}")
    kpi3.metric("Fraud rate", f"{fraud_rate:.2f}%")
    kpi4.metric("Model ROC-AUC", f"{bundle.metrics['roc_auc']:.3f}")

    tab_monitoring, tab_prediction, tab_model = st.tabs(
        ["Monitoring", "Predict", "Model Evidence"]
    )

    with tab_monitoring:
        chart_col, table_col = st.columns([1, 1])

        with chart_col:
            st.subheader("Fraud by Channel")
            channel_fraud = (
                filtered.groupby("channel")["fraud_label"].mean().sort_values()
            )
            fig, ax = plt.subplots(figsize=(7, 4))
            channel_fraud.plot(kind="barh", ax=ax, color="#2f6f73")
            ax.set_xlabel("Fraud rate")
            ax.set_ylabel("")
            st.pyplot(fig, clear_figure=True)

        with table_col:
            st.subheader("Highest Risk Transactions")
            high_risk = filtered.sort_values(
                ["device_risk_score", "merchant_risk_score", "velocity_24h"],
                ascending=False,
            )
            st.dataframe(
                high_risk.head(15)[
                    [
                        "transaction_id",
                        "channel",
                        "transaction_amount_usd",
                        "txn_country",
                        "device_risk_score",
                        "merchant_risk_score",
                        "velocity_24h",
                        "fraud_label",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )

        st.subheader("Transaction Amount Distribution")
        fig_amount, ax_amount = plt.subplots(figsize=(10, 4))
        sns.histplot(
            data=filtered,
            x="transaction_amount_usd",
            hue="fraud_label",
            bins=40,
            ax=ax_amount,
        )
        ax_amount.set_xlabel("Transaction amount (USD)")
        st.pyplot(fig_amount, clear_figure=True)

    with tab_prediction:
        render_prediction_form(transactions, bundle)

    with tab_model:
        st.subheader("Top 10 Selected Features")
        st.write(
            "Features are ranked with mutual information, then the model is retrained using only the top drivers."
        )
        st.dataframe(
            bundle.feature_importance.head(10),
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("Project Evidence")
        st.markdown(
            """
            - Missing values checked across customers, merchants, and transactions.
            - Duplicate records checked across all raw datasets.
            - Data types validated before feature engineering.
            - Fraud-specific features created for velocity, geography, device risk, merchant risk, and night behaviour.
            - Top-10 features selected before final modelling to keep explanations focused.
            - The review threshold is tuned for alert prioritisation rather than criminal proof.
            """
        )


if __name__ == "__main__":
    main()
