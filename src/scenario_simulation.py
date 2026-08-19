"""
Phase 7: Scenario Simulation Engine
======================================
Simulates macroeconomic stress scenarios by applying shock transfer functions
to applicant feature vectors and re-running the credit risk model. Supports
pre-built scenarios (Baseline, Mild Recession, Severe Recession, Rate Hike,
Job Loss) and custom shock parameters. Projects risk trajectories over 12 months.

Optionally integrates real FRED data when FRED_API_KEY is set.

Run: python src/scenario_simulation.py (for testing)
"""

import joblib
import pandas as pd
import numpy as np
import os
import json
import warnings
warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════════════════
# SHOCK TRANSFER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════
# Each scenario defines deltas for macro variables and their downstream
# effects on applicant-level features. These are SIMPLIFYING ASSUMPTIONS,
# not causal ground truth.
#
# | Scenario         | Unemp Δ | Fed Δ  | Income Effect | Util Effect |
# |------------------|---------|--------|---------------|-------------|
# | Baseline         | 0pp     | 0pp    | 0%            | 0%          |
# | Mild Recession   | +2pp    | -1pp   | -5%           | +10%        |
# | Severe Recession | +5pp    | -2pp   | -15%          | +25%        |
# | Rate Hike        | 0pp     | +2pp   | 0%            | +5%         |
# | Job Loss         | +3pp    | 0pp    | -30%          | +40%        |
# ═══════════════════════════════════════════════════════════════════════════

SCENARIOS = {
    "Baseline": {
        "description": "No economic change — current conditions persist.",
        "unemployment_delta_pp": 0.0,
        "fed_funds_delta_pp": 0.0,
        "income_effect_pct": 0.0,
        "utilization_effect_pct": 0.0,
        "dti_effect_pct": 0.0,
        "int_rate_effect_pp": 0.0,
    },
    "Mild Recession": {
        "description": "Moderate economic downturn with rising unemployment.",
        "unemployment_delta_pp": 2.0,
        "fed_funds_delta_pp": -1.0,
        "income_effect_pct": -5.0,
        "utilization_effect_pct": 10.0,
        "dti_effect_pct": 8.0,
        "int_rate_effect_pp": 0.5,
    },
    "Severe Recession": {
        "description": "Deep recession with significant job losses and credit tightening.",
        "unemployment_delta_pp": 5.0,
        "fed_funds_delta_pp": -2.0,
        "income_effect_pct": -15.0,
        "utilization_effect_pct": 25.0,
        "dti_effect_pct": 20.0,
        "int_rate_effect_pp": 1.5,
    },
    "Rate Hike": {
        "description": "Federal Reserve tightening cycle with rising interest rates.",
        "unemployment_delta_pp": 0.0,
        "fed_funds_delta_pp": 2.0,
        "income_effect_pct": 0.0,
        "utilization_effect_pct": 5.0,
        "dti_effect_pct": 3.0,
        "int_rate_effect_pp": 2.0,
    },
    "Job Loss": {
        "description": "Personal job loss scenario — severe income and utilization shock.",
        "unemployment_delta_pp": 3.0,
        "fed_funds_delta_pp": 0.0,
        "income_effect_pct": -30.0,
        "utilization_effect_pct": 40.0,
        "dti_effect_pct": 35.0,
        "int_rate_effect_pp": 0.0,
    },
}


def apply_shock(applicant_df, scenario_params, intensity=1.0):
    """
    Apply a macroeconomic shock to an applicant's feature vector.

    Parameters:
        applicant_df: DataFrame with 1 row of features
        scenario_params: dict with shock deltas
        intensity: float 0–1 for graduated application (month/12)

    Returns:
        Shocked DataFrame (copy of original with adjustments)
    """
    shocked = applicant_df.copy()

    income_pct = scenario_params.get("income_effect_pct", 0) * intensity / 100.0
    util_pct = scenario_params.get("utilization_effect_pct", 0) * intensity / 100.0
    dti_pct = scenario_params.get("dti_effect_pct", 0) * intensity / 100.0
    rate_pp = scenario_params.get("int_rate_effect_pp", 0) * intensity

    # Apply income shock
    if "annual_inc" in shocked.columns and income_pct != 0:
        shocked["annual_inc"] = shocked["annual_inc"] * (1 + income_pct)
        shocked["annual_inc"] = shocked["annual_inc"].clip(lower=0)

    # Recalculate income-derived ratios
    if "loan_to_income_ratio" in shocked.columns and "annual_inc" in shocked.columns:
        inc = shocked["annual_inc"].values[0]
        if inc > 0 and "loan_amnt" in shocked.columns:
            shocked["loan_to_income_ratio"] = shocked["loan_amnt"] / inc
    if "payment_to_income_ratio" in shocked.columns and "annual_inc" in shocked.columns:
        inc = shocked["annual_inc"].values[0]
        if inc > 0 and "installment" in shocked.columns:
            shocked["payment_to_income_ratio"] = (shocked["installment"] * 12) / inc
    if "revol_bal_to_income" in shocked.columns and "annual_inc" in shocked.columns:
        inc = shocked["annual_inc"].values[0]
        if inc > 0 and "revol_bal" in shocked.columns:
            shocked["revol_bal_to_income"] = shocked["revol_bal"] / inc

    # Apply utilization shock
    if "revol_util" in shocked.columns and util_pct != 0:
        shocked["revol_util"] = shocked["revol_util"] * (1 + util_pct)
        shocked["revol_util"] = shocked["revol_util"].clip(upper=100)

    if "credit_utilization_ratio" in shocked.columns and util_pct != 0:
        shocked["credit_utilization_ratio"] = (
            shocked["credit_utilization_ratio"] * (1 + util_pct)
        )
        shocked["credit_utilization_ratio"] = shocked["credit_utilization_ratio"].clip(upper=1.0)

    # Apply DTI shock
    if "dti" in shocked.columns and dti_pct != 0:
        shocked["dti"] = shocked["dti"] * (1 + dti_pct)

    # Apply interest rate shock
    if "int_rate" in shocked.columns and rate_pp != 0:
        shocked["int_rate"] = shocked["int_rate"] + rate_pp

    # Update interaction features
    if "int_rate_x_dti" in shocked.columns:
        if "int_rate" in shocked.columns and "dti" in shocked.columns:
            shocked["int_rate_x_dti"] = shocked["int_rate"] * shocked["dti"]

    if "rate_burden" in shocked.columns:
        if "int_rate" in shocked.columns and "loan_to_income_ratio" in shocked.columns:
            shocked["rate_burden"] = shocked["int_rate"] * shocked["loan_to_income_ratio"]

    # Update macro features if they exist
    if "macro_unemployment_rate" in shocked.columns:
        shocked["macro_unemployment_rate"] += (
            scenario_params.get("unemployment_delta_pp", 0) * intensity
        )
    if "macro_fed_funds_rate" in shocked.columns:
        shocked["macro_fed_funds_rate"] += (
            scenario_params.get("fed_funds_delta_pp", 0) * intensity
        )

    return shocked


def simulate_scenario(model, applicant_df, scenario_name="Baseline",
                      custom_params=None, months=12):
    """
    Project risk trajectory over N months under a given scenario.

    Parameters:
        model: trained sklearn-compatible model with predict_proba
        applicant_df: DataFrame with 1 row of features (no target)
        scenario_name: key from SCENARIOS dict, or "Custom"
        custom_params: dict with custom shock parameters (overrides scenario_name)
        months: number of months to project

    Returns:
        dict with:
          - scenario: name and description
          - baseline_probability: current default probability
          - trajectory: list of {month, probability, delta} for each month
          - peak_probability: maximum projected probability
          - shock_params: the scenario parameters used
    """
    if custom_params:
        params = custom_params
        scenario_desc = "Custom scenario with user-defined parameters"
    elif scenario_name in SCENARIOS:
        params = SCENARIOS[scenario_name]
        scenario_desc = params.get("description", "")
    else:
        raise ValueError(f"Unknown scenario: {scenario_name}. "
                         f"Available: {list(SCENARIOS.keys())}")

    # Get baseline probability
    baseline_prob = float(model.predict_proba(applicant_df)[:, 1][0])

    # Project over months with graduated intensity
    trajectory = []
    for month in range(1, months + 1):
        # Graduated shock: linearly increases from 0 to full intensity
        intensity = month / months
        shocked_df = apply_shock(applicant_df, params, intensity=intensity)
        prob = float(model.predict_proba(shocked_df)[:, 1][0])

        trajectory.append({
            "month": month,
            "probability": round(prob, 4),
            "delta_from_baseline": round(prob - baseline_prob, 4),
            "intensity_pct": round(intensity * 100, 1),
        })

    peak_prob = max(t["probability"] for t in trajectory)

    return {
        "scenario": scenario_name,
        "description": scenario_desc,
        "baseline_probability": round(baseline_prob, 4),
        "trajectory": trajectory,
        "peak_probability": round(peak_prob, 4),
        "final_probability": trajectory[-1]["probability"],
        "total_risk_increase": round(peak_prob - baseline_prob, 4),
        "shock_params": {k: v for k, v in params.items() if k != "description"},
    }


def simulate_all_scenarios(model, applicant_df, months=12):
    """Run all pre-built scenarios and return comparison results."""
    results = {}
    for name in SCENARIOS:
        results[name] = simulate_scenario(model, applicant_df, name, months=months)
    return results


def fetch_fred_data():
    """
    Fetch real macroeconomic data from FRED.
    Returns a DataFrame with monthly macro indicators, or None if API key not set.
    """
    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        print("  FRED_API_KEY not set — using scenario simulation only")
        return None

    try:
        from fredapi import Fred
        fred = Fred(api_key=api_key)

        series = {
            "UNRATE": "unemployment_rate",
            "CPIAUCSL": "cpi",
            "FEDFUNDS": "fed_funds_rate",
            "MORTGAGE30US": "mortgage_rate_30yr",
        }

        frames = []
        for series_id, col_name in series.items():
            try:
                data = fred.get_series(series_id, observation_start="2007-01-01")
                data = data.resample("MS").mean()  # Monthly, start of month
                data.name = col_name
                frames.append(data)
            except Exception as e:
                print(f"  Warning: Could not fetch {series_id}: {e}")

        if frames:
            macro_df = pd.concat(frames, axis=1).dropna()
            macro_df.index.name = "date"
            os.makedirs("data", exist_ok=True)
            macro_df.to_csv("data/fred_macro_data.csv")
            print(f"  Fetched {len(macro_df)} months of FRED data")
            return macro_df

    except ImportError:
        print("  fredapi not installed — using scenario simulation only")

    return None


if __name__ == "__main__":
    print("=" * 70)
    print("  PHASE 7: SCENARIO SIMULATION ENGINE")
    print("=" * 70)

    # Fetch FRED data if available
    print("\nChecking FRED API...")
    fred_data = fetch_fred_data()

    # Quick test
    print("\nRunning scenario simulation test...")
    model = joblib.load("models/lgbm_model.joblib")
    bg = pd.read_parquet("models/background_data.parquet")
    sample = bg.drop(columns=["target"]).iloc[[0]]

    results = simulate_all_scenarios(model, sample, months=12)

    print("\n  Scenario Comparison:")
    print("  " + "-" * 60)
    print(f"  {'Scenario':<20} {'Baseline':>10} {'Peak':>10} {'Delta Risk':>10}")
    print("  " + "-" * 60)
    for name, r in results.items():
        print(f"  {name:<20} {r['baseline_probability']:>10.2%} "
              f"{r['peak_probability']:>10.2%} "
              f"{r['total_risk_increase']:>+10.2%}")

    # Save shock documentation
    shock_doc = []
    for name, params in SCENARIOS.items():
        shock_doc.append({
            "scenario": name,
            **{k: v for k, v in params.items()},
        })

    os.makedirs("reports", exist_ok=True)
    with open("reports/scenario_shock_functions.json", "w") as f:
        json.dump(shock_doc, f, indent=2)

    print(f"\n  Shock functions saved to reports/scenario_shock_functions.json")
    print("  Done!")
