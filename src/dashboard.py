import shap
from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# -----------------------------
# PAGE CONFIGURATION
# -----------------------------

st.set_page_config(
    page_title="Fraud Alert Dashboard",
    layout="wide"
)

# -----------------------------
# TITLE
# -----------------------------

st.title("🚨 Explainable Fraud Alert Prioritization Dashboard")

st.markdown("""
This dashboard simulates an enterprise fraud monitoring and alert prioritization system.
""")

# -----------------------------
# LOAD DATA
# -----------------------------
transactions = pd.read_csv(
    "data/raw/raw/transactions.csv"
)

# -----------------------------
# SIDEBAR FILTERS
# -----------------------------

st.sidebar.header("Fraud Analysis Filters")

country_filter = st.sidebar.multiselect(
    "Select Transaction Country",
    options=transactions['txn_country'].unique(),
    default=transactions['txn_country'].unique()
)

channel_filter = st.sidebar.multiselect(
    "Select Transaction Channel",
    options=transactions['channel'].unique(),
    default=transactions['channel'].unique()
)

fraud_only = st.sidebar.checkbox(
    "Show Fraud Transactions Only"
)

risk_threshold = st.sidebar.slider(
    "Minimum Device Risk Score",
    min_value=0.0,
    max_value=1.0,
    value=0.0,
    step=0.1
)




# -----------------------------
# APPLY FILTERS
# -----------------------------


filtered_transactions = transactions[
    (transactions['txn_country'].isin(country_filter)) &
    (transactions['channel'].isin(channel_filter)) &
    (transactions['device_risk_score'] >= risk_threshold)
]

if fraud_only:
    filtered_transactions = filtered_transactions[
        filtered_transactions['fraud_label'] == 1
    ]

# -----------------------------
# RISK CLASSIFICATION
# -----------------------------

filtered_transactions['risk_level'] = np.where(
    filtered_transactions['device_risk_score'] > 0.8,
    'High Risk',
    np.where(
        filtered_transactions['device_risk_score'] > 0.5,
        'Medium Risk',
        'Low Risk'
    )
)

# -----------------------------
# TRANSACTION SEARCH
# -----------------------------

st.subheader("Transaction Search")

search_id = st.text_input(
    "Enter Transaction ID"
)

if search_id:
    search_results = filtered_transactions[
        filtered_transactions['transaction_id']
        .astype(str)
        .str.contains(search_id)
    ]

    st.dataframe(search_results)

# -----------------------------
# MODEL PREPARATION
# -----------------------------

model_data = transactions.copy()

categorical_cols = [
    'channel',
    'txn_country'
]

le = LabelEncoder()

for col in categorical_cols:
    model_data[col] = le.fit_transform(
        model_data[col]
    )

features = [
    'channel',
    'transaction_amount_usd',
    'txn_country',
    'txn_hour',
    'device_risk_score',
    'new_device_flag',
    'velocity_1h',
    'velocity_24h',
    'geo_distance_km',
    'merchant_risk_score',
    'is_night_flag',
    'alert_generated'
]

X = model_data[features]

y = model_data['fraud_label']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

xgb_model = XGBClassifier(
    n_estimators=50,
    max_depth=4,
    learning_rate=0.1,
    random_state=42
)

xgb_model.fit(X_train, y_train)

explainer = shap.TreeExplainer(xgb_model)

shap_values = explainer.shap_values(X_test)



# -----------------------------
# KPI METRICS
# -----------------------------

total_transactions = len(filtered_transactions)

total_fraud = filtered_transactions['fraud_label'].sum()

fraud_rate = (
    total_fraud / total_transactions
) * 100

col1, col2, col3 = st.columns(3)

col1.metric(
    "Total Transactions",
    f"{total_transactions:,}"
)

col2.metric(
    "Fraudulent Transactions",
    f"{total_fraud:,}"
)

col3.metric(
    "Fraud Rate (%)",
    f"{fraud_rate:.2f}%"
)

# -----------------------------
# RISK LEVEL DISTRIBUTION
# -----------------------------


st.subheader("Transaction Risk Levels")

risk_counts = (
    filtered_transactions['risk_level']
    .value_counts()
)

if len(risk_counts) > 0:

    fig_risk, ax_risk = plt.subplots(figsize=(6,4))

    risk_counts.plot(
        kind='bar',
        ax=ax_risk
    )

    st.pyplot(fig_risk)

else:
    st.warning("No transactions match the selected filters.")


# -----------------------------
# FRAUD DISTRIBUTION
# -----------------------------

st.subheader("Fraud Distribution")

fig, ax = plt.subplots(figsize=(6,4))

sns.countplot(
    x='fraud_label',
    data=filtered_transactions,
    ax=ax
)

st.pyplot(fig)

# -----------------------------
# HIGH RISK TRANSACTIONS
# -----------------------------

st.subheader("High Risk Transactions")

high_risk = filtered_transactions[
    filtered_transactions['device_risk_score'] > 0.7
]

st.dataframe(
    high_risk.head(20)
)

# -----------------------------
# TRANSACTION AMOUNT DISTRIBUTION
# -----------------------------

st.subheader("Transaction Amount Distribution")

fig2, ax2 = plt.subplots(figsize=(10,5))

sns.histplot(
    filtered_transactions['transaction_amount_usd'],
    bins=50,
    ax=ax2
)

st.pyplot(fig2)

# -----------------------------
# FRAUD BY CHANNEL
# -----------------------------

st.subheader("Fraud by Transaction Channel")

channel_fraud = (
    transactions.groupby('channel')['fraud_label']
    .mean()
    .sort_values(ascending=False)
)

fig3, ax3 = plt.subplots(figsize=(8,5))

channel_fraud.plot(
    kind='bar',
    ax=ax3
)

st.pyplot(fig3)

# -----------------------------
# TOP SUSPICIOUS TRANSACTIONS
# -----------------------------

st.subheader("Top Suspicious Transactions")

top_suspicious = filtered_transactions.sort_values(
    by='device_risk_score',
    ascending=False
)

st.dataframe(
    top_suspicious.head(10)
)

# -----------------------------
# SHAP EXPLAINABILITY
# -----------------------------

st.subheader("Explainable AI - Fraud Prediction Drivers")

selected_index = st.slider(
    "Select Transaction Index",
    0,
    len(X_test) - 1,
    0
)

fig_shap, ax_shap = plt.subplots(figsize=(10,5))

shap.waterfall_plot(
    shap.Explanation(
        values=shap_values[selected_index],
        base_values=explainer.expected_value,
        data=X_test.iloc[selected_index],
        feature_names=features
    ),
    show=False
)

st.pyplot(fig_shap)



# -----------------------------
# ANALYST INSIGHTS
# -----------------------------

st.subheader("Analyst Insights")

st.info("""
Key fraud indicators identified:
- High transaction velocity
- Elevated device risk score
- Geographic anomalies
- High merchant risk score
- Night-time transaction behavior
""")