# Screenshot Checklist

Use this checklist before final GitHub submission.

## Required Screenshots

Save updated screenshots in the `screenshots/` folder.

### 1. Dashboard Command Center

Suggested filename:

```text
screenshots/dashboard_overview.png
```

Capture:

- command center KPIs
- tabs visible
- sidebar triage controls

### 2. Alert Queue

Suggested filename:

```text
screenshots/alert_queue.png
```

Capture:

- prioritized transactions
- fraud probability column
- priority tier
- SLA hours

### 3. Investigation Reason Codes

Suggested filename:

```text
screenshots/explainability_reason_codes.png
```

Capture:

- selected transaction
- prediction
- fraud probability
- reason codes
- recommended actions

### 4. Model Governance

Suggested filename:

```text
screenshots/model_governance.png
```

Capture:

- top features
- model comparison
- threshold tuning

## How To Capture

1. Run the app:

```bash
streamlit run src/dashboard.py
```

2. Open:

```text
http://localhost:8501
```

3. Use Windows screenshot shortcut:

```text
Win + Shift + S
```

4. Save each image into the `screenshots/` folder.

## Final Check

After replacing screenshots, confirm they appear correctly in `README.md` on GitHub.
