"""Streamlit fraud analyst workspace.

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
    engineer_features,
    load_transactions,
    predict_transaction,
    train_top_feature_model,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


@st.cache_data(show_spinner=False)
def get_transactions() -> pd.DataFrame:
    """Load the raw transaction dataset once for dashboard use."""

    return load_transactions(REPO_ROOT)


@st.cache_resource(show_spinner="Training fraud prioritization model...")
def get_model_bundle(transactions: pd.DataFrame):
    """Train and cache the top-feature model for dashboard predictions."""

    return train_top_feature_model(transactions, top_n=10)


@st.cache_data(show_spinner=False)
def score_transactions(transactions: pd.DataFrame, top_features: tuple[str, ...]) -> pd.DataFrame:
    """Create the feature matrix used by the cached model scoring step."""

    engineered = engineer_features(transactions)
    return engineered[list(top_features)]


def binary_label(value: int) -> str:
    """Display binary indicators as analyst-friendly text."""

    return "Yes" if int(value) == 1 else "No"


def apply_workspace_style() -> None:
    """Apply a compact analyst-workspace visual treatment."""

    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1420px;
        }
        div[data-testid="stMetric"] {
            border: 1px solid #d8dee7;
            border-left: 4px solid #265f73;
            border-radius: 6px;
            padding: 0.75rem 0.9rem;
            background: #fbfcfd;
        }
        div[data-testid="stMetricValue"] {
            font-size: 1.7rem;
        }
        .workspace-title {
            font-size: 2.05rem;
            line-height: 1.15;
            font-weight: 750;
            color: #17202a;
            margin-bottom: 0.2rem;
        }
        .workspace-subtitle {
            color: #52616f;
            font-size: 0.98rem;
            margin-bottom: 1.2rem;
        }
        .status-high {
            color: #8f1d1d;
            font-weight: 700;
        }
        .status-medium {
            color: #8a5a00;
            font-weight: 700;
        }
        .status-low {
            color: #1f5f46;
            font-weight: 700;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def enrich_alert_queue(transactions: pd.DataFrame, bundle) -> pd.DataFrame:
    """Score every transaction and add triage fields for the alert queue."""

    scored_features = score_transactions(transactions, tuple(bundle.top_features))
    alert_queue = transactions.copy()
    alert_queue["fraud_probability"] = bundle.model.predict_proba(scored_features)[:, 1]
    alert_queue["fraud_probability_pct"] = alert_queue["fraud_probability"] * 100
    alert_queue["priority_score"] = (
        alert_queue["fraud_probability"] * 0.70
        + alert_queue["device_risk_score"] * 0.15
        + alert_queue["merchant_risk_score"] * 0.10
        + (alert_queue["velocity_24h"] / max(alert_queue["velocity_24h"].max(), 1)) * 0.05
    )

    critical_cutoff = min(bundle.review_threshold + 0.25, 0.95)
    review_cutoff = bundle.review_threshold

    alert_queue["priority_tier"] = pd.cut(
        alert_queue["fraud_probability"],
        bins=[-0.01, review_cutoff, critical_cutoff, 1.01],
        labels=["Monitor", "Review", "Critical"],
    )
    alert_queue["priority_tier"] = alert_queue["priority_tier"].astype(str)
    alert_queue["sla_hours"] = alert_queue["priority_tier"].map(
        {"Critical": 4, "Review": 12, "Monitor": 48}
    )

    return alert_queue.sort_values(
        ["priority_score", "fraud_probability"],
        ascending=False,
    ).reset_index(drop=True)


def filter_alert_queue(alert_queue: pd.DataFrame) -> pd.DataFrame:
    """Apply sidebar filters to the scored alert queue."""

    with st.sidebar:
        st.header("Triage Controls")
        countries = sorted(alert_queue["txn_country"].dropna().unique())
        channels = sorted(alert_queue["channel"].dropna().unique())
        tiers = ["Critical", "Review", "Monitor"]

        country_filter = st.multiselect("Country", countries, default=countries)
        channel_filter = st.multiselect("Channel", channels, default=channels)
        tier_filter = st.multiselect("Priority", tiers, default=tiers)
        min_probability = st.slider("Minimum fraud probability", 0.0, 1.0, 0.0, 0.05)
        show_confirmed_fraud = st.checkbox("Confirmed fraud only")

    filtered = alert_queue[
        alert_queue["txn_country"].isin(country_filter)
        & alert_queue["channel"].isin(channel_filter)
        & alert_queue["priority_tier"].isin(tier_filter)
        & (alert_queue["fraud_probability"] >= min_probability)
    ].copy()

    if show_confirmed_fraud:
        filtered = filtered[filtered["fraud_label"] == 1]

    return filtered


def render_header(bundle) -> None:
    """Render the dashboard header."""

    st.markdown(
        """
        <div class="workspace-title">Fraud Alert Prioritization Workspace</div>
        <div class="workspace-subtitle">
        Explainable transaction risk scoring, triage queue management, and analyst decision support.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        f"Selected model: {bundle.model_name} | Review threshold: {bundle.review_threshold:.2f}"
    )


def render_command_center(filtered: pd.DataFrame, bundle) -> None:
    """Render executive triage KPIs."""

    total_alerts = len(filtered)
    confirmed_fraud = int(filtered["fraud_label"].sum()) if total_alerts else 0
    fraud_rate = confirmed_fraud / total_alerts if total_alerts else 0
    critical_alerts = int((filtered["priority_tier"] == "Critical").sum())
    avg_probability = filtered["fraud_probability"].mean() if total_alerts else 0

    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    kpi1.metric("Queue Volume", f"{total_alerts:,}")
    kpi2.metric("Critical Alerts", f"{critical_alerts:,}")
    kpi3.metric("Confirmed Fraud", f"{confirmed_fraud:,}")
    kpi4.metric("Fraud Rate", f"{fraud_rate:.2%}")
    kpi5.metric("Avg Risk", f"{avg_probability:.1%}")

    st.caption(
        f"Model ROC-AUC {bundle.metrics['roc_auc']:.3f} | "
        f"Precision {bundle.metrics['precision']:.3f} | "
        f"Recall {bundle.metrics['recall']:.3f}"
    )


def render_overview(filtered: pd.DataFrame) -> None:
    """Render portfolio-level monitoring charts."""

    left, middle, right = st.columns([1, 1, 1])

    with left:
        st.subheader("Priority Mix")
        tier_order = ["Critical", "Review", "Monitor"]
        tier_counts = filtered["priority_tier"].value_counts().reindex(tier_order).fillna(0)
        fig, ax = plt.subplots(figsize=(6, 4))
        tier_counts.plot(kind="bar", ax=ax, color=["#8f1d1d", "#c8892b", "#2f6f73"])
        ax.set_xlabel("")
        ax.set_ylabel("Transactions")
        ax.set_title("")
        st.pyplot(fig, clear_figure=True)

    with middle:
        st.subheader("Risk By Channel")
        channel_risk = filtered.groupby("channel")["fraud_probability"].mean().sort_values()
        fig, ax = plt.subplots(figsize=(6, 4))
        channel_risk.plot(kind="barh", ax=ax, color="#3b82a0")
        ax.set_xlabel("Average fraud probability")
        ax.set_ylabel("")
        ax.set_xlim(0, max(float(channel_risk.max()) * 1.15, 0.05) if len(channel_risk) else 1)
        st.pyplot(fig, clear_figure=True)

    with right:
        st.subheader("Risk By Country")
        country_risk = (
            filtered.groupby("txn_country")["fraud_probability"]
            .mean()
            .sort_values(ascending=False)
            .head(8)
        )
        fig, ax = plt.subplots(figsize=(6, 4))
        country_risk.plot(kind="bar", ax=ax, color="#637381")
        ax.set_xlabel("")
        ax.set_ylabel("Average fraud probability")
        st.pyplot(fig, clear_figure=True)


def render_alert_queue(filtered: pd.DataFrame) -> None:
    """Render the triage table."""

    st.subheader("Prioritized Alert Queue")
    queue_columns = [
        "transaction_id",
        "priority_tier",
        "sla_hours",
        "fraud_probability_pct",
        "priority_score",
        "channel",
        "transaction_amount_usd",
        "txn_country",
        "txn_hour",
        "device_risk_score",
        "merchant_risk_score",
        "velocity_24h",
        "fraud_label",
    ]
    st.dataframe(
        filtered.head(40)[queue_columns],
        use_container_width=True,
        hide_index=True,
        column_config={
            "fraud_probability_pct": st.column_config.ProgressColumn(
                "Fraud Probability",
                format="%.1f%%",
                min_value=0,
                max_value=100,
            ),
            "priority_score": st.column_config.NumberColumn(
                "Priority Score",
                format="%.3f",
            ),
            "transaction_amount_usd": st.column_config.NumberColumn(
                "Amount USD",
                format="$%.2f",
            ),
        },
    )


def render_investigation(filtered: pd.DataFrame, transactions: pd.DataFrame, bundle) -> None:
    """Render a selected transaction investigation workspace."""

    st.subheader("Transaction Investigation")

    if filtered.empty:
        st.warning("No transactions match the current filters.")
        return

    transaction_options = filtered["transaction_id"].head(80).tolist()
    selected_transaction = st.selectbox("Transaction ID", transaction_options)
    selected_row = filtered.loc[filtered["transaction_id"] == selected_transaction].iloc[0]

    transaction_payload = {
        "channel": selected_row["channel"],
        "transaction_amount_usd": float(selected_row["transaction_amount_usd"]),
        "txn_country": selected_row["txn_country"],
        "txn_hour": int(selected_row["txn_hour"]),
        "device_risk_score": float(selected_row["device_risk_score"]),
        "new_device_flag": int(selected_row["new_device_flag"]),
        "velocity_1h": int(selected_row["velocity_1h"]),
        "velocity_24h": int(selected_row["velocity_24h"]),
        "geo_distance_km": float(selected_row["geo_distance_km"]),
        "merchant_risk_score": float(selected_row["merchant_risk_score"]),
        "is_night_flag": int(selected_row["is_night_flag"]),
        "alert_generated": int(selected_row["alert_generated"]),
    }
    fraud_probability, label, drivers = predict_transaction(
        bundle,
        transaction_payload,
        transactions,
    )

    summary_col, details_col = st.columns([1, 2])

    with summary_col:
        st.metric("Prediction", label)
        st.metric("Fraud Probability", f"{fraud_probability:.1%}")
        st.metric("Priority Tier", selected_row["priority_tier"])
        st.metric("SLA Hours", int(selected_row["sla_hours"]))

        if label == "Fraudulent":
            st.error("Escalate for manual fraud review.")
        else:
            st.success("Monitor unless additional risk evidence appears.")

    with details_col:
        st.dataframe(
            pd.DataFrame(
                [
                    ["Amount", f"${selected_row['transaction_amount_usd']:,.2f}"],
                    ["Channel", selected_row["channel"]],
                    ["Country", selected_row["txn_country"]],
                    ["Hour", int(selected_row["txn_hour"])],
                    ["Device risk", f"{selected_row['device_risk_score']:.3f}"],
                    ["Merchant risk", f"{selected_row['merchant_risk_score']:.3f}"],
                    ["Velocity 24h", int(selected_row["velocity_24h"])],
                    ["Geo distance km", f"{selected_row['geo_distance_km']:,.1f}"],
                ],
                columns=["Signal", "Value"],
            ),
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("Reason Codes")
    reason_columns = [
        "feature",
        "input_value",
        "reference_value",
        "probability_impact",
        "severity",
        "risk_direction",
        "recommended_action",
    ]
    st.dataframe(
        drivers[reason_columns],
        use_container_width=True,
        hide_index=True,
        column_config={
            "probability_impact": st.column_config.NumberColumn(
                "Probability Impact",
                format="%+.2f",
            )
        },
    )


def render_scenario_testing(transactions: pd.DataFrame, bundle) -> None:
    """Render the manual transaction scenario form."""

    st.subheader("Scenario Testing")

    with st.form("transaction_prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            channel = st.selectbox("Channel", sorted(transactions["channel"].dropna().unique()))
            amount = st.number_input(
                "Amount USD",
                min_value=0.0,
                value=float(transactions["transaction_amount_usd"].median()),
                step=10.0,
            )
            country = st.selectbox("Country", sorted(transactions["txn_country"].dropna().unique()))
            txn_hour = st.slider("Transaction Hour", 0, 23, 12)

        with col2:
            device_risk = st.slider("Device Risk", 0.0, 1.0, 0.55, 0.01)
            merchant_risk = st.slider("Merchant Risk", 0.0, 1.0, 0.45, 0.01)
            geo_distance = st.number_input(
                "Geo Distance KM",
                min_value=0.0,
                value=float(transactions["geo_distance_km"].median()),
                step=25.0,
            )
            velocity_24h = st.number_input(
                "Velocity 24h",
                min_value=0,
                value=int(transactions["velocity_24h"].median()),
                step=1,
            )

        with col3:
            velocity_1h = st.number_input(
                "Velocity 1h",
                min_value=0,
                value=int(transactions["velocity_1h"].median()),
                step=1,
            )
            new_device = st.selectbox("New Device", [0, 1], format_func=binary_label)
            is_night = st.selectbox("Night Transaction", [0, 1], format_func=binary_label)
            alert_generated = st.selectbox(
                "Existing Rule Alert",
                [0, 1],
                format_func=binary_label,
            )

        submitted = st.form_submit_button("Score Transaction", type="primary")

    if not submitted:
        return

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

    fraud_probability, label, drivers = predict_transaction(bundle, transaction, transactions)
    top_driver = drivers.iloc[0]

    score_col, reason_col = st.columns([1, 2])
    with score_col:
        st.metric("Prediction", label)
        st.metric("Fraud Probability", f"{fraud_probability:.1%}")
        if fraud_probability >= bundle.review_threshold:
            st.error("Manual review recommended.")
        else:
            st.success("Below current review threshold.")

    with reason_col:
        st.markdown(
            f"""
            **Primary driver:** `{top_driver['feature']}`  
            **Probability impact:** `{top_driver['probability_impact']:+.1%}`  
            **Severity:** `{top_driver['severity']}`  
            **Recommended action:** {top_driver['recommended_action']}
            """
        )

    st.dataframe(
        drivers[
            [
                "feature",
                "input_value",
                "reference_value",
                "probability_impact",
                "severity",
                "risk_direction",
                "analyst_reason",
                "recommended_action",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )


def render_model_governance(bundle) -> None:
    """Render model evidence and governance details."""

    feature_col, model_col = st.columns([1, 1])

    with feature_col:
        st.subheader("Top Fraud Drivers")
        st.dataframe(
            bundle.feature_importance.head(10),
            use_container_width=True,
            hide_index=True,
        )

    with model_col:
        st.subheader("Model Comparison")
        st.dataframe(
            bundle.model_comparison,
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("Threshold Tuning")
    st.dataframe(
        bundle.threshold_table.sort_values("f1", ascending=False).head(10),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Governance Notes")
    st.markdown(
        """
        - Raw data quality is checked before modelling.
        - Fraud-specific features are engineered from analyst-relevant signals.
        - The model is restricted to the top selected features for interpretability.
        - Threshold tuning balances detection quality with alert volume.
        - Local explanations provide reason codes and investigation actions.
        """
    )


def main() -> None:
    """Build the fraud monitoring and investigation workspace."""

    st.set_page_config(
        page_title="Fraud Alert Prioritization Workspace",
        layout="wide",
    )
    apply_workspace_style()

    transactions = get_transactions()
    bundle = get_model_bundle(transactions)
    alert_queue = enrich_alert_queue(transactions, bundle)
    filtered = filter_alert_queue(alert_queue)

    render_header(bundle)
    render_command_center(filtered, bundle)

    tab_overview, tab_queue, tab_investigation, tab_scenario, tab_governance = st.tabs(
        [
            "Command Center",
            "Alert Queue",
            "Investigation",
            "Scenario Testing",
            "Model Governance",
        ]
    )

    with tab_overview:
        render_overview(filtered)

    with tab_queue:
        render_alert_queue(filtered)

    with tab_investigation:
        render_investigation(filtered, transactions, bundle)

    with tab_scenario:
        render_scenario_testing(transactions, bundle)

    with tab_governance:
        render_model_governance(bundle)


if __name__ == "__main__":
    main()
