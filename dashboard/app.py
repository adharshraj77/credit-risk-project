"""
Phase 9: Interactive Credit Risk Dashboard
=============================================
Multi-tab Streamlit dashboard with Plotly charts, SHAP explanations,
counterfactual recommendations, scenario simulation, and feature store drift.

Run: streamlit run dashboard/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib
import json
import os
import sys
import shap
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# Add project root to path for imports and set working directory
# This ensures models/, reports/, src/ paths work on Streamlit Cloud
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

# ═══════════════════════════════════════════════════════════════════════════
# PAGE CONFIG & CUSTOM CSS
# ═══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Credit Risk Intelligence",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* Global */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
.main .block-container {
    padding-top: 1.5rem;
    max-width: 1400px;
}

/* Metric Cards */
.metric-card {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border: 1px solid rgba(99, 102, 241, 0.2);
    border-radius: 16px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    backdrop-filter: blur(8px);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 40px rgba(99, 102, 241, 0.2);
}
.metric-label {
    color: #a0aec0;
    font-size: 0.8rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.3rem;
}
.metric-value {
    font-size: 1.8rem;
    font-weight: 700;
    background: linear-gradient(135deg, #667eea, #764ba2);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.metric-value.good { background: linear-gradient(135deg, #48bb78, #38a169); -webkit-background-clip: text; }
.metric-value.warn { background: linear-gradient(135deg, #ed8936, #dd6b20); -webkit-background-clip: text; }
.metric-value.bad  { background: linear-gradient(135deg, #fc8181, #e53e3e); -webkit-background-clip: text; }

/* Risk Badge */
.risk-badge {
    display: inline-block;
    padding: 0.5rem 1.5rem;
    border-radius: 50px;
    font-weight: 700;
    font-size: 1rem;
    letter-spacing: 0.05em;
}
.risk-low    { background: rgba(72, 187, 120, 0.15); color: #48bb78; border: 1px solid #48bb78; }
.risk-medium { background: rgba(236, 201, 75, 0.15); color: #ecc94b; border: 1px solid #ecc94b; }
.risk-high   { background: rgba(237, 137, 54, 0.15); color: #ed8936; border: 1px solid #ed8936; }
.risk-vhigh  { background: rgba(252, 129, 129, 0.15); color: #fc8181; border: 1px solid #fc8181; }

/* Section Headers */
.section-header {
    font-size: 1.1rem;
    font-weight: 600;
    color: #e2e8f0;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid rgba(99, 102, 241, 0.3);
    margin-bottom: 1rem;
}

/* Summary Box */
.summary-box {
    background: linear-gradient(135deg, #1a1a2e, #0f3460);
    border: 1px solid rgba(99, 102, 241, 0.15);
    border-radius: 12px;
    padding: 1.5rem;
    margin: 1rem 0;
    line-height: 1.8;
}

/* Hide Streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px 8px 0 0;
    padding: 8px 20px;
    font-weight: 500;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# PLOTLY THEME
# ═══════════════════════════════════════════════════════════════════════════

PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#e2e8f0"),
    margin=dict(l=40, r=40, t=50, b=40),
)
COLORS = {
    "primary": "#667eea",
    "secondary": "#764ba2",
    "good": "#48bb78",
    "bad": "#fc8181",
    "warn": "#ed8936",
    "accent": "#4fd1c5",
    "gradient": ["#667eea", "#764ba2", "#4fd1c5", "#48bb78", "#ed8936"],
}


def metric_card(label, value, css_class=""):
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value {css_class}">{value}</div>
    </div>
    """


def risk_badge(probability):
    if probability < 0.15:
        return '<span class="risk-badge risk-low">🟢 LOW RISK</span>'
    elif probability < 0.35:
        return '<span class="risk-badge risk-medium">🟡 MEDIUM RISK</span>'
    elif probability < 0.55:
        return '<span class="risk-badge risk-high">🟠 HIGH RISK</span>'
    else:
        return '<span class="risk-badge risk-vhigh">🔴 VERY HIGH RISK</span>'


# ═══════════════════════════════════════════════════════════════════════════
# AUTO-PIPELINE: Run training if artifacts are missing or corrupted
# ═══════════════════════════════════════════════════════════════════════════

def _check_artifacts_ok():
    """Return True if all required model artifacts can be loaded."""
    required_joblib = [
        "models/lgbm_model.joblib",
        "models/feature_names.joblib",
    ]
    for path in required_joblib:
        if not os.path.exists(path):
            return False
    # Try loading the parquet – catches pyarrow version mismatches
    try:
        pd.read_parquet("models/background_data.parquet")
    except Exception:
        return False
    return True


def _run_pipeline():
    """Run the full pipeline (clean → engineer → feature store → train → explain)."""
    import subprocess, sys

    python = sys.executable  # use the same python that is running streamlit

    steps = [
        ("Phase 1: Data Cleaning", [python, "src/data_cleaning.py"]),
        ("Phase 2: Feature Engineering", [python, "src/feature_engineering.py"]),
        ("Phase 3: Feature Store", [python, "src/feature_store.py"]),
        ("Phase 4: Model Training", [python, "src/model_training.py"]),
        ("Phase 5: Explainability", [python, "src/explainability.py"]),
    ]

    # If processed features already exist, skip phases 1-3
    if os.path.exists("data/processed/features.parquet"):
        steps = [s for s in steps if "Phase 4" in s[0] or "Phase 5" in s[0]]

    progress = st.progress(0, text="Initializing pipeline...")
    for i, (label, cmd) in enumerate(steps):
        progress.progress((i) / len(steps), text=f"Running {label}...")
        result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if result.returncode != 0:
            st.error(f"{label} failed:\n```\n{result.stderr[-2000:]}\n```")
            st.stop()
    progress.progress(1.0, text="Pipeline complete!")


# ═══════════════════════════════════════════════════════════════════════════
# LOAD ARTIFACTS
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_artifacts():
    """Load all model artifacts."""
    model = joblib.load("models/lgbm_model.joblib")
    features = joblib.load("models/feature_names.joblib")
    background = pd.read_parquet("models/background_data.parquet")

    # Load calibrated model if available, else use raw
    cal_path = "models/calibrated_model.joblib"
    calibrated = joblib.load(cal_path) if os.path.exists(cal_path) else model

    # Load metrics
    metrics = {}
    if os.path.exists("models/metrics.json"):
        with open("models/metrics.json") as f:
            metrics = json.load(f)

    # Load SHAP explainer
    explainer = None
    if os.path.exists("models/shap_explainer.joblib"):
        explainer = joblib.load("models/shap_explainer.joblib")

    # Load feature types for DiCE
    feature_types = None
    if os.path.exists("models/dice_feature_types.joblib"):
        feature_types = joblib.load("models/dice_feature_types.joblib")

    return model, calibrated, features, background, metrics, explainer, feature_types


# Auto-run pipeline if artifacts are missing / corrupted
if not _check_artifacts_ok():
    # Check if we have the raw data or processed features to run the pipeline
    has_raw = any(os.path.isfile(p) for p in [
        "data/raw/accepted_2007_to_2018Q4.csv",
        "data/raw/accepted_2007_to_2018q4.csv/accepted_2007_to_2018Q4.csv",
        "data/raw/accepted_2007_to_2018Q4.csv.gz",
    ])
    has_processed = os.path.exists("data/processed/features.parquet")

    if has_raw or has_processed:
        st.warning("⚙️ Model artifacts are missing or incompatible. Auto-running the training pipeline...")
        _run_pipeline()
        st.cache_resource.clear()
        st.rerun()
    else:
        st.error("Could not load model artifacts and no training data found.")
        st.info("Download the data first (`python download_data.py`), then restart the dashboard.")
        st.stop()

try:
    model, calibrated_model, feature_names, background_data, metrics, shap_explainer, feature_types = load_artifacts()
except Exception as e:
    st.error(f"Could not load model artifacts: {e}")
    st.info("Run the training pipeline first: `python src/model_training.py` then `python src/explainability.py`")
    st.stop()


# ═══════════════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════════════

st.markdown("""
<div style="text-align: center; padding: 1rem 0;">
    <h1 style="font-size: 2rem; font-weight: 800;
        background: linear-gradient(135deg, #667eea, #764ba2, #4fd1c5);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
        🏦 Credit Risk Intelligence Platform
    </h1>
    <p style="color: #a0aec0; font-size: 0.95rem; margin-top: -0.5rem;">
        Time-Aware ML Model with Explainable & Counterfactual Decision Support
    </p>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Model Performance",
    "🔍 Individual Assessment",
    "🔄 Counterfactuals",
    "🌍 Scenario Simulation",
    "📈 Feature Store & Drift",
])

# ═══════════════════════════════════════════════════════════════════════════
# TAB 1: MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════

with tab1:
    if not metrics:
        st.warning("No metrics.json found. Run model training first.")
    else:
        perf = metrics.get("performance", {})

        # ── Key Metrics Cards ────────────────────────────────────────────
        cols = st.columns(5)
        with cols[0]:
            auc_val = perf.get("roc_auc", "N/A")
            css = "good" if isinstance(auc_val, (int, float)) and auc_val > 0.7 else ""
            st.markdown(metric_card("ROC AUC", f"{auc_val:.4f}" if isinstance(auc_val, float) else auc_val, css), unsafe_allow_html=True)
        with cols[1]:
            pr_val = perf.get("pr_auc", perf.get("average_precision", "N/A"))
            st.markdown(metric_card("PR AUC", f"{pr_val:.4f}" if isinstance(pr_val, float) else pr_val), unsafe_allow_html=True)
        with cols[2]:
            ks_val = perf.get("ks_statistic", "N/A")
            st.markdown(metric_card("KS Statistic", f"{ks_val:.4f}" if isinstance(ks_val, float) else ks_val), unsafe_allow_html=True)
        with cols[3]:
            rec_val = perf.get("default", {}).get("recall", "N/A")
            st.markdown(metric_card("Default Recall", f"{rec_val:.4f}" if isinstance(rec_val, float) else rec_val), unsafe_allow_html=True)
        with cols[4]:
            acc_val = perf.get("accuracy", "N/A")
            st.markdown(metric_card("Accuracy", f"{acc_val:.4f}" if isinstance(acc_val, float) else acc_val), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        charts = metrics.get("charts", {})

        # ── ROC & PR Curves ──────────────────────────────────────────────
        col_roc, col_pr = st.columns(2)

        with col_roc:
            roc = charts.get("roc_curve", {})
            if roc:
                fig_roc = go.Figure()
                fig_roc.add_trace(go.Scatter(
                    x=roc["fpr"], y=roc["tpr"],
                    mode="lines",
                    name=f"LightGBM (AUC={perf.get('roc_auc', 0):.3f})",
                    line=dict(color=COLORS["primary"], width=2.5),
                    fill="tozeroy",
                    fillcolor="rgba(102,126,234,0.1)",
                ))
                fig_roc.add_trace(go.Scatter(
                    x=[0, 1], y=[0, 1],
                    mode="lines", name="Random",
                    line=dict(color="rgba(255,255,255,0.2)", dash="dash", width=1),
                ))
                fig_roc.update_layout(
                    title="ROC Curve",
                    xaxis_title="False Positive Rate",
                    yaxis_title="True Positive Rate",
                    **PLOTLY_LAYOUT,
                    legend=dict(x=0.55, y=0.1),
                    height=400,
                )
                st.plotly_chart(fig_roc, use_container_width=True)

        with col_pr:
            pr = charts.get("pr_curve", {})
            if pr:
                fig_pr = go.Figure()
                fig_pr.add_trace(go.Scatter(
                    x=pr["recall"], y=pr["precision"],
                    mode="lines",
                    name=f"LightGBM (AP={perf.get('pr_auc', perf.get('average_precision', 0)):.3f})",
                    line=dict(color=COLORS["accent"], width=2.5),
                    fill="tozeroy",
                    fillcolor="rgba(79,209,197,0.1)",
                ))
                fig_pr.update_layout(
                    title="Precision-Recall Curve",
                    xaxis_title="Recall",
                    yaxis_title="Precision",
                    **PLOTLY_LAYOUT,
                    legend=dict(x=0.05, y=0.1),
                    height=400,
                )
                st.plotly_chart(fig_pr, use_container_width=True)

        # ── Confusion Matrix & Calibration ───────────────────────────────
        col_cm, col_cal = st.columns(2)

        with col_cm:
            cm = charts.get("confusion_matrix")
            if cm:
                cm_arr = np.array(cm)
                fig_cm = go.Figure(data=go.Heatmap(
                    z=cm_arr,
                    x=["Predicted Good", "Predicted Default"],
                    y=["Actual Good", "Actual Default"],
                    text=[[f"{v:,}" for v in row] for row in cm_arr],
                    texttemplate="%{text}",
                    textfont=dict(size=16, color="white"),
                    colorscale=[[0, "#1a1a2e"], [1, "#667eea"]],
                    showscale=False,
                ))
                fig_cm.update_layout(
                    title="Confusion Matrix",
                    **PLOTLY_LAYOUT,
                    height=400,
                )
                st.plotly_chart(fig_cm, use_container_width=True)

        with col_cal:
            cal = charts.get("calibration_curve", {})
            if cal:
                fig_cal = go.Figure()
                fig_cal.add_trace(go.Scatter(
                    x=cal["mean_predicted_value"],
                    y=cal["fraction_of_positives"],
                    mode="lines+markers",
                    name="Calibrated Model",
                    line=dict(color=COLORS["primary"], width=2.5),
                    marker=dict(size=8),
                ))
                fig_cal.add_trace(go.Scatter(
                    x=[0, 1], y=[0, 1],
                    mode="lines", name="Perfectly Calibrated",
                    line=dict(color="rgba(255,255,255,0.2)", dash="dash"),
                ))
                fig_cal.update_layout(
                    title="Calibration Curve",
                    xaxis_title="Mean Predicted Probability",
                    yaxis_title="Fraction of Positives",
                    **PLOTLY_LAYOUT,
                    height=400,
                )
                st.plotly_chart(fig_cal, use_container_width=True)

        # ── Feature Importance & Score Distribution ──────────────────────
        col_fi, col_sd = st.columns(2)

        with col_fi:
            fi = charts.get("feature_importance", [])
            if fi:
                fi_df = pd.DataFrame(fi).head(20)
                fig_fi = go.Figure(go.Bar(
                    x=fi_df["importance"],
                    y=fi_df["feature"],
                    orientation="h",
                    marker=dict(
                        color=fi_df["importance"],
                        colorscale=[[0, "#4fd1c5"], [0.5, "#667eea"], [1, "#764ba2"]],
                    ),
                ))
                fig_fi.update_layout(
                    title="Top 20 Feature Importance (Gain)",
                    xaxis_title="Importance",
                    yaxis=dict(autorange="reversed"),
                    **PLOTLY_LAYOUT,
                    height=550,
                )
                st.plotly_chart(fig_fi, use_container_width=True)

        with col_sd:
            sd = charts.get("score_distribution", {})
            if sd:
                fig_sd = go.Figure()
                fig_sd.add_trace(go.Bar(
                    x=sd["bins"], y=sd["good_loans"],
                    name="Good Loans",
                    marker_color="rgba(72, 187, 120, 0.7)",
                ))
                fig_sd.add_trace(go.Bar(
                    x=sd["bins"], y=sd["defaults"],
                    name="Defaults",
                    marker_color="rgba(252, 129, 129, 0.7)",
                ))
                fig_sd.update_layout(
                    title="Score Distribution",
                    xaxis_title="Predicted Default Probability",
                    yaxis_title="Count",
                    barmode="overlay",
                    **PLOTLY_LAYOUT,
                    height=550,
                )
                st.plotly_chart(fig_sd, use_container_width=True)

        # ── Model Comparison Table ───────────────────────────────────────
        with st.expander("📋 Model Details & Comparison"):
            comp_cols = st.columns(2)
            with comp_cols[0]:
                st.markdown("**Dataset Info**")
                ds = metrics.get("dataset", {})
                st.markdown(f"""
                | Property | Value |
                |----------|-------|
                | Total Rows | {ds.get('total_rows', 'N/A'):,} |
                | Features | {ds.get('num_features', 'N/A')} |
                | Train Size | {ds.get('train_size', 'N/A'):,} |
                | Val Size | {ds.get('val_size', 'N/A'):,} |
                | Test Size | {ds.get('test_size', 'N/A'):,} |
                | Train Default Rate | {ds.get('train_default_rate', 0):.2%} |
                | Test Default Rate | {ds.get('test_default_rate', 0):.2%} |
                """)
            with comp_cols[1]:
                st.markdown("**Model Comparison**")
                baseline = metrics.get("baseline_performance", {})
                st.markdown(f"""
                | Model | ROC AUC |
                |-------|---------|
                | Logistic Regression | {baseline.get('logreg_roc_auc', 'N/A')} |
                | LightGBM (Calibrated) | {perf.get('roc_auc', 'N/A')} |
                """)
                hp = metrics.get("hyperparameters", {})
                if hp:
                    st.markdown("**LightGBM Hyperparameters**")
                    st.json(hp)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2: INDIVIDUAL ASSESSMENT
# ═══════════════════════════════════════════════════════════════════════════

with tab2:
    st.markdown('<div class="section-header">🔍 Individual Risk Assessment</div>', unsafe_allow_html=True)

    # Sidebar-like controls in columns
    ctrl_col, result_col = st.columns([1, 3])

    X_bg = background_data.drop(columns=["target"], errors="ignore")

    with ctrl_col:
        st.markdown("**Select Applicant**")
        sample_idx = st.selectbox(
            "Background sample index",
            options=list(range(len(X_bg))),
            format_func=lambda x: f"Applicant #{x}",
            key="applicant_select",
        )
        applicant = X_bg.iloc[[sample_idx]].copy()

        st.markdown("**Override Features**")
        override_features = {}
        for feat in ["loan_amnt", "annual_inc", "int_rate", "dti", "revol_util"]:
            if feat in applicant.columns:
                current_val = float(applicant[feat].values[0])
                if feat == "loan_amnt":
                    new_val = st.number_input(f"Loan Amount ($)", min_value=0.0, max_value=2000000.0, value=current_val, step=1000.0, format="%.2f", key=f"ov_{feat}")
                elif feat == "annual_inc":
                    new_val = st.number_input(f"Annual Income ($)", min_value=1.0, max_value=10000000.0, value=max(1.0, current_val), step=1000.0, format="%.2f", key=f"ov_{feat}")
                elif feat == "int_rate":
                    new_val = st.number_input(f"Interest Rate (%)", min_value=0.0, max_value=100.0, value=current_val, step=0.1, format="%.2f", key=f"ov_{feat}")
                elif feat == "dti":
                    new_val = st.number_input(f"DTI Ratio", min_value=0.0, max_value=500.0, value=current_val, step=1.0, format="%.2f", key=f"ov_{feat}")
                elif feat == "revol_util":
                    new_val = st.number_input(f"Revolving Util (%)", min_value=0.0, max_value=300.0, value=current_val, step=1.0, format="%.2f", key=f"ov_{feat}")
                else:
                    new_val = current_val
                applicant[feat] = new_val
                override_features[feat] = new_val

        # Recalculate derived features
        if "loan_to_income_ratio" in applicant.columns and "annual_inc" in applicant.columns:
            inc = applicant["annual_inc"].values[0]
            if inc > 0 and "loan_amnt" in applicant.columns:
                applicant["loan_to_income_ratio"] = applicant["loan_amnt"].values[0] / inc

    with result_col:
        # Predict
        try:
            prob = float(calibrated_model.predict_proba(applicant)[:, 1][0])
        except Exception:
            prob = float(model.predict_proba(applicant)[:, 1][0])

        # Risk gauge and badge
        gauge_col, badge_col = st.columns([2, 1])

        with gauge_col:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                number={"suffix": "%", "font": {"size": 40}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1},
                    "bar": {"color": "#667eea"},
                    "bgcolor": "rgba(0,0,0,0)",
                    "steps": [
                        {"range": [0, 15], "color": "rgba(72,187,120,0.2)"},
                        {"range": [15, 35], "color": "rgba(236,201,75,0.2)"},
                        {"range": [35, 55], "color": "rgba(237,137,54,0.2)"},
                        {"range": [55, 100], "color": "rgba(252,129,129,0.2)"},
                    ],
                    "threshold": {
                        "line": {"color": "white", "width": 2},
                        "thickness": 0.8,
                        "value": prob * 100,
                    },
                },
                title={"text": "Default Probability", "font": {"size": 16}},
            ))
            fig_gauge.update_layout(
                height=280,
                **{k: v for k, v in PLOTLY_LAYOUT.items() if k != "margin"},
                margin=dict(l=30, r=30, t=60, b=20),
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

        with badge_col:
            st.markdown("<br><br>", unsafe_allow_html=True)
            st.markdown(risk_badge(prob), unsafe_allow_html=True)
            st.markdown(f"<br>**Probability**: {prob:.2%}", unsafe_allow_html=True)

            threshold = metrics.get("performance", {}).get("optimal_threshold", 0.5)
            if prob > threshold:
                st.error(f"⛔ Exceeds threshold ({threshold:.2%})")
            else:
                st.success(f"✅ Below threshold ({threshold:.2%})")

        # SHAP Explanation
        st.markdown('<div class="section-header">Why This Score? (SHAP Analysis)</div>', unsafe_allow_html=True)

        if shap_explainer is not None:
            try:
                shap_vals = shap_explainer.shap_values(applicant)
                if isinstance(shap_vals, list):
                    sv = shap_vals[1]
                    base_val = shap_explainer.expected_value[1]
                else:
                    sv = shap_vals
                    base_val = shap_explainer.expected_value

                feat_names = feature_names if len(feature_names) == sv.shape[1] else applicant.columns.tolist()

                # Top drivers table
                abs_sv = np.abs(sv[0])
                top_idx = np.argsort(abs_sv)[::-1][:10]

                drivers_data = []
                for idx in top_idx:
                    drivers_data.append({
                        "Feature": feat_names[idx],
                        "Value": f"{applicant.iloc[0, idx]:.2f}" if isinstance(applicant.iloc[0, idx], (int, float, np.floating)) else str(applicant.iloc[0, idx]),
                        "SHAP Impact": f"{sv[0][idx]:+.4f}",
                        "Direction": "🔴 ↑ Risk" if sv[0][idx] > 0 else "🟢 ↓ Risk",
                    })

                driver_col, waterfall_col = st.columns([1, 2])
                with driver_col:
                    st.markdown("**Top 10 Risk Drivers**")
                    st.dataframe(pd.DataFrame(drivers_data), hide_index=True, use_container_width=True)

                with waterfall_col:
                    # SHAP waterfall plot
                    shap_explanation = shap.Explanation(
                        values=sv[0],
                        base_values=base_val,
                        data=applicant.values[0],
                        feature_names=feat_names,
                    )
                    fig_shap, ax = plt.subplots(figsize=(10, 6))
                    plt.style.use("dark_background")
                    shap.plots.waterfall(shap_explanation, max_display=12, show=False)
                    fig_shap.patch.set_facecolor("#0e1117")
                    ax.set_facecolor("#0e1117")
                    st.pyplot(fig_shap)
                    plt.close(fig_shap)
            except Exception as e:
                st.error(f"SHAP analysis error: {e}")
        else:
            st.info("SHAP explainer not available. Run `python src/explainability.py`")

        # Decision Summary
        try:
            from src.decision_summary import generate_risk_assessment
            if shap_explainer is not None:
                top_drivers = []
                for idx in top_idx[:5]:
                    top_drivers.append({
                        "feature": feat_names[idx],
                        "shap_value": float(sv[0][idx]),
                        "direction": "increases risk" if sv[0][idx] > 0 else "decreases risk",
                        "feature_value": float(applicant.iloc[0, idx]) if isinstance(applicant.iloc[0, idx], (int, float, np.floating)) else None,
                    })
                with st.expander("📝 AI Decision Summary", expanded=False):
                    summary = generate_risk_assessment(prob, top_drivers)
                    st.markdown(summary)
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════════
# TAB 3: COUNTERFACTUALS
# ═══════════════════════════════════════════════════════════════════════════

with tab3:
    st.markdown('<div class="section-header">🔄 Counterfactual Recommendations</div>', unsafe_allow_html=True)
    st.markdown(
        "What **minimal changes** would move this applicant to a lower risk tier? "
        "Only actionable features (income, loan amount, etc.) are varied — "
        "historical facts like credit history length are held fixed."
    )

    applicant_cf = X_bg.iloc[[sample_idx]].copy()
    # Apply overrides
    for feat, val in override_features.items():
        if feat in applicant_cf.columns:
            applicant_cf[feat] = val

    try:
        prob_cf = float(calibrated_model.predict_proba(applicant_cf)[:, 1][0])
    except Exception:
        prob_cf = float(model.predict_proba(applicant_cf)[:, 1][0])

    st.markdown(f"**Current default probability**: {prob_cf:.2%}")

    if prob_cf <= 0.15:
        st.success("✅ This applicant is already **Low Risk**. No counterfactuals needed.")
    else:
        if st.button("🔄 Generate Counterfactuals", key="gen_cf"):
            with st.spinner("Finding minimal paths to approval..."):
                try:
                    from src.counterfactual_engine import generate_counterfactuals, build_dice_explainer
                    dice_exp, ftv = build_dice_explainer(model, background_data, feature_types)
                    result = generate_counterfactuals(
                        applicant_cf, n_counterfactuals=3,
                        desired_class=0, dice_exp=dice_exp,
                        features_to_vary=ftv,
                    )

                    if result["success"] and result["counterfactuals_df"] is not None:
                        cf_df = result["counterfactuals_df"]
                        changes = result["changes"]

                        st.markdown("### Suggested Alternatives")

                        for i, change_set in enumerate(changes):
                            if not change_set:
                                continue
                            with st.expander(f"Option {i+1}", expanded=(i == 0)):
                                change_data = []
                                for feat, delta in change_set.items():
                                    if isinstance(delta, dict) and "original" in delta:
                                        orig = delta.get("original", "")
                                        sugg = delta.get("suggested", "")
                                        chg = delta.get("change_pct", 0)
                                        change_data.append({
                                            "Feature": feat,
                                            "Current": f"{orig:,.2f}" if isinstance(orig, (int, float)) else str(orig),
                                            "Suggested": f"{sugg:,.2f}" if isinstance(sugg, (int, float)) else str(sugg),
                                            "Change": f"{chg:+.1f}%" if isinstance(chg, (int, float)) else "",
                                        })
                                if change_data:
                                    st.dataframe(pd.DataFrame(change_data), hide_index=True, use_container_width=True)

                        # Show full CF dataframe
                        with st.expander("Raw Counterfactual Data"):
                            st.dataframe(cf_df)
                    else:
                        st.warning("Could not find viable counterfactuals. Try adjusting the applicant's features.")
                        if "error" in result:
                            st.caption(f"Error: {result['error']}")
                except ImportError:
                    st.error("Counterfactual engine not available. Ensure `src/counterfactual_engine.py` exists.")
                except Exception as e:
                    st.error(f"Counterfactual generation failed: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 4: SCENARIO SIMULATION
# ═══════════════════════════════════════════════════════════════════════════

with tab4:
    st.markdown('<div class="section-header">🌍 Macroeconomic Scenario Simulation</div>', unsafe_allow_html=True)
    st.markdown(
        "Stress-test this applicant under different macroeconomic scenarios. "
        "Shock transfer functions adjust income, utilization, DTI, and interest rates."
    )

    try:
        from src.scenario_simulation import SCENARIOS, simulate_scenario, simulate_all_scenarios

        sim_applicant = X_bg.iloc[[sample_idx]].copy()
        for feat, val in override_features.items():
            if feat in sim_applicant.columns:
                sim_applicant[feat] = val

        # Scenario selection
        sc_col1, sc_col2 = st.columns([1, 2])

        with sc_col1:
            st.markdown("**Pre-built Scenarios**")
            selected_scenarios = st.multiselect(
                "Select scenarios to simulate",
                options=list(SCENARIOS.keys()),
                default=["Baseline", "Mild Recession", "Severe Recession"],
                key="scenario_select",
            )

            st.markdown("---")
            st.markdown("**Custom Scenario**")
            use_custom = st.checkbox("Add custom scenario", key="use_custom")
            if use_custom:
                custom_params = {
                    "unemployment_delta_pp": st.slider("Unemployment Δ (pp)", -5.0, 10.0, 2.0, 0.5),
                    "fed_funds_delta_pp": st.slider("Fed Funds Δ (pp)", -5.0, 5.0, 0.0, 0.5),
                    "income_effect_pct": st.slider("Income Effect (%)", -50.0, 20.0, -10.0, 1.0),
                    "utilization_effect_pct": st.slider("Utilization Effect (%)", -20.0, 80.0, 15.0, 1.0),
                    "dti_effect_pct": st.slider("DTI Effect (%)", -20.0, 50.0, 10.0, 1.0),
                    "int_rate_effect_pp": st.slider("Int Rate Δ (pp)", -3.0, 5.0, 0.5, 0.25),
                }

        with sc_col2:
            if st.button("🚀 Run Simulation", key="run_sim"):
                with st.spinner("Simulating scenarios..."):
                    results = {}
                    use_model = calibrated_model if calibrated_model is not None else model

                    for sc_name in selected_scenarios:
                        results[sc_name] = simulate_scenario(
                            use_model, sim_applicant, sc_name, months=12
                        )

                    if use_custom:
                        results["Custom"] = simulate_scenario(
                            use_model, sim_applicant, "Custom",
                            custom_params=custom_params, months=12
                        )

                    # Trajectory chart
                    fig_traj = go.Figure()
                    colors = COLORS["gradient"]
                    for i, (name, res) in enumerate(results.items()):
                        traj = res["trajectory"]
                        fig_traj.add_trace(go.Scatter(
                            x=[t["month"] for t in traj],
                            y=[t["probability"] * 100 for t in traj],
                            mode="lines+markers",
                            name=name,
                            line=dict(color=colors[i % len(colors)], width=2.5),
                            marker=dict(size=6),
                        ))
                    # Add baseline line
                    baseline_prob = list(results.values())[0]["baseline_probability"]
                    fig_traj.add_hline(
                        y=baseline_prob * 100,
                        line_dash="dash",
                        line_color="rgba(255,255,255,0.3)",
                        annotation_text="Current Risk",
                    )
                    fig_traj.update_layout(
                        title="12-Month Risk Trajectory Under Stress Scenarios",
                        xaxis_title="Month",
                        yaxis_title="Default Probability (%)",
                        **PLOTLY_LAYOUT,
                        height=450,
                    )
                    st.plotly_chart(fig_traj, use_container_width=True)

                    # Comparison bar chart
                    bar_names = list(results.keys())
                    bar_baseline = [results[n]["baseline_probability"] * 100 for n in bar_names]
                    bar_peak = [results[n]["peak_probability"] * 100 for n in bar_names]

                    fig_bar = go.Figure()
                    fig_bar.add_trace(go.Bar(
                        x=bar_names, y=bar_baseline,
                        name="Baseline",
                        marker_color="rgba(102,126,234,0.7)",
                    ))
                    fig_bar.add_trace(go.Bar(
                        x=bar_names, y=bar_peak,
                        name="Peak (12mo)",
                        marker_color="rgba(252,129,129,0.7)",
                    ))
                    fig_bar.update_layout(
                        title="Scenario Comparison: Baseline vs Peak Risk",
                        yaxis_title="Default Probability (%)",
                        barmode="group",
                        **PLOTLY_LAYOUT,
                        height=400,
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)

                    # Summary table
                    st.markdown("### Scenario Summary")
                    summary_data = []
                    for name, res in results.items():
                        summary_data.append({
                            "Scenario": name,
                            "Baseline": f"{res['baseline_probability']:.2%}",
                            "Peak Risk": f"{res['peak_probability']:.2%}",
                            "Final (12mo)": f"{res['final_probability']:.2%}",
                            "Risk Increase": f"{res['total_risk_increase']:+.2%}",
                        })
                    st.dataframe(pd.DataFrame(summary_data), hide_index=True, use_container_width=True)

        # Shock Function Documentation
        with st.expander("📋 Shock Transfer Function Documentation"):
            st.markdown("""
            | Scenario | Unemployment Δ | Fed Funds Δ | Income Effect | Utilization Effect | DTI Effect | Rate Δ |
            |----------|---------------|-------------|---------------|-------------------|------------|--------|
            | Baseline | 0pp | 0pp | 0% | 0% | 0% | 0pp |
            | Mild Recession | +2pp | -1pp | -5% | +10% | +8% | +0.5pp |
            | Severe Recession | +5pp | -2pp | -15% | +25% | +20% | +1.5pp |
            | Rate Hike | 0pp | +2pp | 0% | +5% | +3% | +2pp |
            | Job Loss | +3pp | 0pp | -30% | +40% | +35% | 0pp |

            **Note**: These are simplifying assumptions for stress-testing, not causal relationships.
            Shocks are applied gradually over 12 months (linear ramp from 0% to 100% intensity).
            """)
    except ImportError:
        st.error("Scenario simulation module not found. Ensure `src/scenario_simulation.py` exists.")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 5: FEATURE STORE & DRIFT
# ═══════════════════════════════════════════════════════════════════════════

with tab5:
    st.markdown('<div class="section-header">📈 Feature Store & Data Drift</div>', unsafe_allow_html=True)

    try:
        from src.feature_store import get_snapshot_summary, get_drift_metrics

        # Monthly portfolio summary
        st.markdown("### Portfolio Over Time")
        snapshots = get_snapshot_summary()

        if not snapshots.empty:
            fig_port = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                subplot_titles=("Monthly Loan Volume", "Monthly Default Rate"),
                vertical_spacing=0.12,
            )
            fig_port.add_trace(go.Bar(
                x=snapshots["snapshot_date"],
                y=snapshots["n_loans"],
                marker_color="rgba(102,126,234,0.6)",
                name="Loans",
            ), row=1, col=1)
            fig_port.add_trace(go.Scatter(
                x=snapshots["snapshot_date"],
                y=snapshots["default_rate"] * 100,
                mode="lines+markers",
                line=dict(color=COLORS["bad"], width=2),
                marker=dict(size=4),
                name="Default Rate",
            ), row=2, col=1)
            fig_port.update_layout(
                **PLOTLY_LAYOUT,
                height=500,
                showlegend=False,
            )
            fig_port.update_yaxes(title_text="Count", row=1, col=1)
            fig_port.update_yaxes(title_text="Default Rate (%)", row=2, col=1)
            st.plotly_chart(fig_port, use_container_width=True)

        # PSI Drift
        st.markdown("### Population Stability Index (PSI) Drift")
        drift = get_drift_metrics()

        if not drift.empty:
            psi_cols = [c for c in drift.columns if c.startswith("psi_") and c != "psi_overall"]
            fig_drift = go.Figure()
            fig_drift.add_trace(go.Scatter(
                x=drift["snapshot_date"],
                y=drift["psi_overall"],
                mode="lines+markers",
                name="Overall PSI",
                line=dict(color=COLORS["primary"], width=2.5),
                marker=dict(size=6),
            ))
            # Threshold lines
            fig_drift.add_hline(y=0.1, line_dash="dash",
                              line_color="rgba(236,201,75,0.5)",
                              annotation_text="Moderate (0.1)")
            fig_drift.add_hline(y=0.2, line_dash="dash",
                              line_color="rgba(252,129,129,0.5)",
                              annotation_text="Significant (0.2)")
            fig_drift.update_layout(
                title="Monthly PSI Drift",
                xaxis_title="Month",
                yaxis_title="PSI",
                **PLOTLY_LAYOUT,
                height=400,
            )
            st.plotly_chart(fig_drift, use_container_width=True)

            # Per-feature drift
            with st.expander("Per-Feature PSI Details"):
                st.dataframe(drift, use_container_width=True)

            st.markdown("""
            **PSI Interpretation:**
            - **< 0.1**: No significant change — model is stable
            - **0.1 – 0.2**: Moderate change — investigate
            - **> 0.2**: Significant change — retrain recommended
            """)
        else:
            st.info("No drift metrics available. Run `python src/feature_store.py` first.")

    except ImportError:
        st.info("Feature store module not found. Run `python src/feature_store.py` to build it.")
    except Exception as e:
        st.warning(f"Could not load feature store data: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown(
    '<div style="text-align: center; color: #4a5568; font-size: 0.8rem;">'
    '🏦 Credit Risk Intelligence Platform | '
    'Built with Streamlit, LightGBM, SHAP, DiCE | '
    'Time-Aware ML with Explainable Decision Support'
    '</div>',
    unsafe_allow_html=True,
)
