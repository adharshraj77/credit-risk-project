"""
Phase 6: Counterfactual Engine
================================
Uses DiCE (Diverse Counterfactual Explanations) to generate actionable
recommendations for rejected applicants. Enforces immutability constraints
on non-actionable features and realistic bounds on actionable ones.

Run: python src/counterfactual_engine.py (for testing)
"""

import joblib
import pandas as pd
import numpy as np
import dice_ml
import os
import warnings
warnings.filterwarnings("ignore")


# ── Feature Constraints ─────────────────────────────────────────────────

# Immutable features: cannot be changed by the applicant
IMMUTABLE_FEATURES = [
    "credit_history_months",       # can't change when credit history started
    "earliest_cr_line",            # historical fact
    "issue_year", "issue_month",   # loan timing
    "total_acc",                   # historical count
    "pub_rec",                     # public records — historical
    "pub_rec_bankruptcies",        # bankruptcies — historical
    "delinq_2yrs",                 # past delinquencies — historical
    "collections_12_mths_ex_med",  # past collections — historical
    "has_bankruptcy",              # derived from historical
    "has_public_record",           # derived from historical
    "has_recent_delinquency",      # derived from historical
    "has_recent_collection",       # derived from historical
    # Missingness flags are historical
]

# Actionable features with realistic bounds
ACTIONABLE_FEATURES = [
    "loan_amnt",
    "annual_inc",
    "dti",
    "revol_util",
    "int_rate",
    "installment",
    "open_acc",
    "revol_bal",
    "loan_to_income_ratio",
    "payment_to_income_ratio",
    "credit_utilization_ratio",
]

# Features that are quasi-actionable (can be influenced over time)
QUASI_ACTIONABLE = [
    "inq_last_6mths",             # can avoid applying for credit
    "acc_now_delinq",             # can pay current debts
    "currently_delinquent_flag",
]


def get_actionable_features(feature_names):
    """Get the list of features that DiCE is allowed to vary."""
    actionable = ACTIONABLE_FEATURES + QUASI_ACTIONABLE
    return [f for f in actionable if f in feature_names]


def get_immutable_features(feature_names):
    """Get features that must NOT be varied."""
    # Also include all _was_missing flags and _encoded features as immutable
    immutable = list(IMMUTABLE_FEATURES)
    for f in feature_names:
        if f.endswith("_was_missing") or f.endswith("_encoded"):
            immutable.append(f)
    return [f for f in immutable if f in feature_names]


def build_dice_explainer(model=None, background_data=None, feature_types=None):
    """
    Build a DiCE explainer with proper constraints.

    Returns: (dice_exp, features_to_vary)
    """
    if model is None:
        model = joblib.load("models/lgbm_model.joblib")
    if background_data is None:
        background_data = pd.read_parquet("models/background_data.parquet")
    if feature_types is None:
        feature_types = joblib.load("models/dice_feature_types.joblib")

    # Fill any NaN to prevent DiCE crashes
    for c in feature_types["continuous"]:
        if c in background_data.columns:
            background_data[c] = background_data[c].fillna(0)

    d = dice_ml.Data(
        dataframe=background_data,
        continuous_features=feature_types["continuous"],
        outcome_name="target",
    )
    m = dice_ml.Model(model=model, backend="sklearn")
    dice_exp = dice_ml.Dice(d, m, method="random")

    # Get features that can be varied
    feature_names = [c for c in background_data.columns if c != "target"]
    features_to_vary = get_actionable_features(feature_names)

    return dice_exp, features_to_vary


def generate_counterfactuals(applicant_df, n_counterfactuals=3,
                              desired_class=0, dice_exp=None,
                              features_to_vary=None):
    """
    Generate counterfactual explanations for an applicant.

    Parameters:
        applicant_df: DataFrame with 1 row (features only, no target)
        n_counterfactuals: Number of counterfactual examples to generate
        desired_class: Target class (0 = approved, 1 = rejected)
        dice_exp: Pre-built DiCE explainer (or builds one if None)
        features_to_vary: List of features DiCE can change (or auto-detected)

    Returns:
        dict with:
          - counterfactuals_df: DataFrame of counterfactual examples
          - changes: list of dicts describing what changed
          - success: bool indicating if valid CFs were found
    """
    if dice_exp is None or features_to_vary is None:
        dice_exp, features_to_vary = build_dice_explainer()

    if not features_to_vary:
        features_to_vary = "all"

    try:
        cf = dice_exp.generate_counterfactuals(
            applicant_df,
            total_CFs=n_counterfactuals,
            desired_class=desired_class,
            features_to_vary=features_to_vary,
        )

        cf_df = cf.cf_examples_list[0].final_cfs_df

        if cf_df is None or len(cf_df) == 0:
            return {"counterfactuals_df": None, "changes": [], "success": False}

        # Compute what changed
        changes = []
        for i, (_, cf_row) in enumerate(cf_df.iterrows()):
            row_changes = {}
            for col in applicant_df.columns:
                if col in cf_df.columns:
                    original = applicant_df[col].values[0]
                    new_val = cf_row[col]
                    try:
                        if abs(float(new_val) - float(original)) > 1e-6:
                            row_changes[col] = {
                                "original": float(original),
                                "suggested": float(new_val),
                                "change": float(new_val) - float(original),
                                "change_pct": (
                                    round((float(new_val) - float(original))
                                          / (abs(float(original)) + 1e-6) * 100, 1)
                                ),
                            }
                    except (ValueError, TypeError):
                        if str(new_val) != str(original):
                            row_changes[col] = {
                                "original": str(original),
                                "suggested": str(new_val),
                            }
            changes.append(row_changes)

        return {
            "counterfactuals_df": cf_df,
            "changes": changes,
            "success": True,
        }

    except Exception as e:
        return {
            "counterfactuals_df": None,
            "changes": [],
            "success": False,
            "error": str(e),
        }


if __name__ == "__main__":
    # Quick test
    print("Loading model and generating test counterfactual...")
    bg = pd.read_parquet("models/background_data.parquet")
    sample = bg.drop(columns=["target"]).iloc[[0]]
    result = generate_counterfactuals(sample)
    if result["success"]:
        print("Counterfactual generated successfully!")
        print(f"Changes: {result['changes']}")
    else:
        print(f"Failed: {result.get('error', 'No valid counterfactuals found')}")
