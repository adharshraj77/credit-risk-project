"""
Phase 5: Explainability — SHAP
================================
Builds SHAP TreeExplainer for global feature importance and local per-applicant
explanations. Provides a function to extract top-N risk drivers for any applicant.

Run: python src/explainability.py
"""

import joblib
import pandas as pd
import numpy as np
import shap
import os
import time
import json


def build_explainers():
    """Build SHAP explainer and compute global importance."""

    model_path = "models/lgbm_model.joblib"
    background_path = "models/background_data.parquet"

    if not os.path.exists(model_path):
        print(f"ERROR: Could not find model at {model_path}")
        return

    print("=" * 70)
    print("  PHASE 5: EXPLAINABILITY (SHAP)")
    print("=" * 70)

    start = time.time()

    # ── Load artifacts ───────────────────────────────────────────────────
    print("\n[1/4] Loading model and background data...")
    model = joblib.load(model_path)
    background_data = pd.read_parquet(background_path)
    features = joblib.load("models/feature_names.joblib")

    X_background = background_data.drop(columns=["target"], errors="ignore")
    print(f"  Background samples: {len(X_background)}")
    print(f"  Features: {len(features)}")

    # ── Build SHAP TreeExplainer ─────────────────────────────────────────
    print("\n[2/4] Building SHAP TreeExplainer...")
    explainer = shap.TreeExplainer(model)
    joblib.dump(explainer, "models/shap_explainer.joblib")
    print("  TreeExplainer built and saved")

    # ── Compute global SHAP values on background data ────────────────────
    print("\n[3/4] Computing global SHAP values on background data...")
    shap_values = explainer.shap_values(X_background)

    # For binary classification, shap_values may be a list [class_0, class_1]
    if isinstance(shap_values, list):
        # Use class 1 (default) SHAP values
        sv = shap_values[1]
    else:
        sv = shap_values

    # Global feature importance (mean absolute SHAP value)
    global_importance = pd.DataFrame({
        "feature": features if len(features) == sv.shape[1] else X_background.columns.tolist(),
        "mean_abs_shap": np.abs(sv).mean(axis=0),
        "mean_shap": sv.mean(axis=0),
        "std_shap": sv.std(axis=0),
    }).sort_values("mean_abs_shap", ascending=False)

    # Save global importance
    global_importance.to_csv("reports/shap_global_importance.csv", index=False)

    # Save SHAP values for background data
    np.save("models/shap_values_background.npy", sv)

    top10 = global_importance.head(10)
    print("\n  Top 10 Global Risk Drivers:")
    print("  " + "-" * 50)
    for _, row in top10.iterrows():
        bar = "█" * int(row["mean_abs_shap"] / top10["mean_abs_shap"].max() * 20)
        print(f"  {row['feature']:<35} {row['mean_abs_shap']:.4f}  {bar}")

    # ── Prepare DiCE metadata ────────────────────────────────────────────
    print("\n[4/4] Preparing counterfactual metadata...")

    cat_features = X_background.select_dtypes(
        include=["category", "object"]
    ).columns.tolist()
    cont_features = X_background.select_dtypes(
        exclude=["category", "object"]
    ).columns.tolist()

    feature_types = {
        "continuous": cont_features,
        "categorical": cat_features,
    }
    joblib.dump(feature_types, "models/dice_feature_types.joblib")

    elapsed = time.time() - start

    print(f"\n{'=' * 70}")
    print(f"  EXPLAINABILITY COMPLETE")
    print(f"  SHAP explainer: models/shap_explainer.joblib")
    print(f"  Global importance: reports/shap_global_importance.csv")
    print(f"  Background SHAP values: models/shap_values_background.npy")
    print(f"  Time: {elapsed:.1f}s")
    print(f"{'=' * 70}")


def explain_applicant(applicant_df, top_n=5):
    """
    Generate local SHAP explanation for a single applicant.

    Parameters:
        applicant_df: DataFrame with one row (features only, no target)
        top_n: Number of top risk drivers to return

    Returns:
        dict with:
          - shap_values: raw SHAP values for the applicant
          - top_drivers: list of (feature, shap_value, direction, feature_value)
          - base_value: expected value (base prediction)
    """
    explainer = joblib.load("models/shap_explainer.joblib")
    features = joblib.load("models/feature_names.joblib")

    shap_vals = explainer.shap_values(applicant_df)

    # Handle list output for binary classification
    if isinstance(shap_vals, list):
        sv = shap_vals[1][0]  # class 1, first (only) sample
        base_value = explainer.expected_value[1]
    else:
        sv = shap_vals[0]
        base_value = explainer.expected_value

    # Get feature names
    feat_names = features if len(features) == len(sv) else applicant_df.columns.tolist()

    # Sort by absolute SHAP value
    indices = np.argsort(np.abs(sv))[::-1][:top_n]

    top_drivers = []
    for idx in indices:
        feat_name = feat_names[idx]
        shap_val = float(sv[idx])
        direction = "increases risk" if shap_val > 0 else "decreases risk"
        feat_val = float(applicant_df.iloc[0, idx]) if idx < applicant_df.shape[1] else None
        top_drivers.append({
            "feature": feat_name,
            "shap_value": round(shap_val, 4),
            "direction": direction,
            "feature_value": feat_val,
        })

    return {
        "shap_values": sv.tolist(),
        "top_drivers": top_drivers,
        "base_value": float(base_value),
    }


if __name__ == "__main__":
    build_explainers()
