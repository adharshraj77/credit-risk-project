# Time-Aware Credit Risk Model

A portfolio-grade ML project with Explainable and Counterfactual Decision Support.

## Architecture

```
┌─────────────────────────────────────┐
│           DATA SOURCES              │
│  Lending Club (loans) + FRED (macro)│
└───────────────────┬─────────────────┘
                    │
                    ▼
      Data Cleaning Pipeline (P1)
 (leakage removal, imputation, target def)
                    │
                    ▼
    Feature Engineering Layer (P2)
 (DTI, utilization, delinquencies, ratios)
                    │
                    ▼
  Time-Versioned Feature Store (P3)
 (monthly snapshots · DuckDB · PSI drift)
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
  Current Feature       Historical
     Vector              Timeline
          │                   │
          └─────────┬─────────┘
                    ▼
      Credit Risk Model (P4)
 (LogReg → LightGBM · Isotonic Calibration)
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
  SHAP Explainability   Counterfactual Engine
  (global + local)      (DiCE · actionable)
          │                   │
          │                   ▼
          │         Scenario Simulation (P7)
          │       (FRED shocks · 12mo trajectory)
          │                   │
          └─────────┬─────────┘
                    ▼
      Decision Summary Layer (P8)
 (templated NL · SHAP + CF + scenario outputs)
                    │
                    ▼
    Interactive Dashboard (P9)
 (5-tab Streamlit · Plotly charts · live scenarios)
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Core | Python 3.11+, pandas, numpy |
| Feature Store | DuckDB + Parquet (time-versioned) |
| Models | LightGBM, Logistic Regression, scikit-learn |
| Calibration | CalibratedClassifierCV (isotonic) |
| Explainability | SHAP (TreeExplainer) |
| Counterfactuals | DiCE (Diverse Counterfactual Explanations) |
| Scenarios | FRED API + custom shock functions |
| Dashboard | Streamlit + Plotly |
| Data Profiling | ydata-profiling |

## Setup

1. **Create virtual environment and install dependencies:**
   ```bash
   python -m venv venv
   .\venv\Scripts\Activate.ps1   # Windows
   # source venv/bin/activate    # Linux/Mac
   pip install -r requirements.txt
   ```

2. **Download data:**
   ```bash
   python download_data.py           # Lending Club
   python download_home_credit.py    # Home Credit (Phase 11)
   ```

3. **Optional — FRED API Key** (for real macro data in scenario simulation):
   ```bash
   set FRED_API_KEY=your_key_here    # Windows
   # export FRED_API_KEY=your_key    # Linux/Mac
   ```
   Get a free key at [fred.stlouisfed.org](https://fred.stlouisfed.org/docs/api/api_key.html).

## Pipeline — Run Order

Execute scripts sequentially from the project root:

```bash
# Phase 1: Data Cleaning
python src/data_cleaning.py

# Phase 2: Feature Engineering (20+ credit risk features)
python src/feature_engineering.py

# Phase 3: Time-Versioned Feature Store + PSI Drift
python src/feature_store.py

# Phase 4: Model Training (LogReg + LightGBM + Calibration)
python src/model_training.py

# Phase 5: SHAP Explainability (global + local)
python src/explainability.py

# Phase 7: Scenario Simulation (FRED + shock functions)
python src/scenario_simulation.py

# Phase 9: Launch Dashboard
streamlit run dashboard/app.py

# Phase 11: Home Credit Validation (optional)
python src/home_credit_pipeline.py
```

## Dashboard

Launch the interactive dashboard:
```bash
streamlit run dashboard/app.py
```

### Dashboard Tabs

| Tab | Description |
|-----|-------------|
| 📊 Model Performance | ROC/PR curves, confusion matrix, calibration, feature importance |
| 🔍 Individual Assessment | Per-applicant risk score, SHAP waterfall, risk drivers |
| 🔄 Counterfactuals | Minimal actionable changes for approval (DiCE) |
| 🌍 Scenario Simulation | Macro stress-testing with 12-month risk trajectories |
| 📈 Feature Store & Drift | Portfolio trends, PSI drift monitoring |

## Model Performance

After training, metrics are saved to `models/metrics.json`. Key results:
- **Temporal split**: Train ≤2015, Validation 2016, Test ≥2017
- **Class imbalance**: Handled via `scale_pos_weight`
- **Calibration**: Isotonic regression on validation set

## Project Structure

```
credit-risk-project/
├── dashboard/
│   └── app.py                    # Streamlit dashboard (Phase 9)
├── data/
│   ├── raw/                      # Raw downloaded data
│   ├── interim/                  # Cleaned data
│   ├── processed/                # Engineered features
│   └── feature_store.duckdb      # Time-versioned store
├── models/
│   ├── lgbm_model.joblib         # Trained LightGBM
│   ├── calibrated_model.joblib   # Calibrated model
│   ├── logreg_model.joblib       # Logistic Regression baseline
│   ├── shap_explainer.joblib     # SHAP TreeExplainer
│   ├── metrics.json              # Comprehensive metrics + chart data
│   └── home_credit/              # Phase 11 validation
├── reports/
│   ├── data_cleaning_notes.md    # Cleaning decisions
│   ├── feature_dictionary.md     # Feature documentation
│   ├── drift_metrics.csv         # PSI drift data
│   ├── validation_report.md      # Cross-dataset comparison
│   └── PROJECT_WRITEUP.md        # Full project documentation
├── src/
│   ├── data_cleaning.py          # Phase 1
│   ├── feature_engineering.py    # Phase 2
│   ├── feature_store.py          # Phase 3
│   ├── model_training.py         # Phase 4
│   ├── explainability.py         # Phase 5
│   ├── counterfactual_engine.py  # Phase 6
│   ├── scenario_simulation.py    # Phase 7
│   ├── decision_summary.py       # Phase 8
│   └── home_credit_pipeline.py   # Phase 11
├── download_data.py              # Lending Club download
├── download_home_credit.py       # Home Credit download
└── requirements.txt
```

## Documentation

See [reports/PROJECT_WRITEUP.md](reports/PROJECT_WRITEUP.md) for the full project write-up, including:
- Problem framing and regulatory context
- Dataset choices and limitations
- Time-based validation rationale
- Leakage analysis
- Calibration results
- Scenario simulation assumptions
- Cross-dataset validation findings
