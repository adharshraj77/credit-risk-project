"""
Phase 2: Feature Engineering
==============================
Builds a comprehensive credit-risk feature set from cleaned Lending Club data
using DuckDB for efficient SQL-based transformations. Includes credit ratios,
delinquency metrics, time-based features, interaction terms, and target encoding
for high-cardinality categoricals.

Run: python src/feature_engineering.py
"""

import duckdb
import pandas as pd
import numpy as np
import os
import time
import json


def engineer_features():
    """Build full feature set from cleaned Lending Club data."""

    interim_path = "data/interim/lending_club_clean.parquet"
    output_path = "data/processed/features.parquet"

    if not os.path.exists(interim_path):
        print(f"ERROR: Could not find {interim_path}")
        print("Run src/data_cleaning.py first.")
        return None

    print("=" * 70)
    print("  PHASE 2: FEATURE ENGINEERING")
    print("=" * 70)

    start = time.time()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # ── Load data ────────────────────────────────────────────────────────
    print("\n[1/5] Loading cleaned data...")
    df = pd.read_parquet(interim_path)
    print(f"  {len(df):,} rows × {len(df.columns)} columns")

    # ── DuckDB feature engineering ───────────────────────────────────────
    print("\n[2/5] Engineering features via DuckDB...")
    con = duckdb.connect(database=":memory:")
    con.register("clean_data", df)

    feature_query = """
    SELECT
        *,

        -- ═══ Time-Based Features ═══
        date_diff('month', earliest_cr_line, issue_d) AS credit_history_months,
        extract('year' FROM issue_d)  AS issue_year,
        extract('month' FROM issue_d) AS issue_month,

        -- ═══ Income / Affordability Ratios ═══
        CASE WHEN annual_inc > 0
             THEN loan_amnt / annual_inc
             ELSE 0
        END AS loan_to_income_ratio,

        CASE WHEN annual_inc > 0
             THEN (installment * 12.0) / annual_inc
             ELSE 0
        END AS payment_to_income_ratio,

        CASE WHEN annual_inc > 0
             THEN revol_bal / annual_inc
             ELSE 0
        END AS revol_bal_to_income,

        -- ═══ Credit Utilization Features ═══
        CASE WHEN total_rev_hi_lim > 0
             THEN revol_bal * 1.0 / total_rev_hi_lim
             ELSE 0
        END AS credit_utilization_ratio,

        CASE WHEN total_rev_hi_lim > 0
             THEN (total_rev_hi_lim - revol_bal) * 1.0 / total_rev_hi_lim
             ELSE 0
        END AS available_credit_pct,

        -- ═══ Account-Level Features ═══
        CASE WHEN total_acc > 0
             THEN open_acc * 1.0 / total_acc
             ELSE 0
        END AS open_acc_ratio,

        CASE WHEN total_acc > 0
             THEN delinq_2yrs * 1.0 / total_acc
             ELSE 0
        END AS delinquency_ratio,

        CASE WHEN open_acc > 0
             THEN inq_last_6mths * 1.0 / open_acc
             ELSE 0
        END AS inquiry_to_open_ratio,

        CASE WHEN total_acc > 0
             THEN pub_rec * 1.0 / total_acc
             ELSE 0
        END AS public_record_ratio,

        CASE WHEN total_acc > 0
             THEN tot_cur_bal * 1.0 / total_acc
             ELSE 0
        END AS avg_balance_per_account,

        -- ═══ Balance Features ═══
        CASE WHEN loan_amnt > 0
             THEN revol_bal * 1.0 / loan_amnt
             ELSE 0
        END AS revolving_to_loan_ratio,

        CASE WHEN tot_hi_cred_lim > 0
             THEN tot_cur_bal * 1.0 / tot_hi_cred_lim
             ELSE 0
        END AS total_utilization,

        -- ═══ Risk Indicator Features ═══
        CASE WHEN acc_now_delinq > 0 THEN 1 ELSE 0 END AS currently_delinquent_flag,

        CASE WHEN pub_rec > 0 THEN 1 ELSE 0 END AS has_public_record,

        CASE WHEN pub_rec_bankruptcies > 0 THEN 1 ELSE 0 END AS has_bankruptcy,

        CASE WHEN delinq_2yrs > 0 THEN 1 ELSE 0 END AS has_recent_delinquency,

        CASE WHEN collections_12_mths_ex_med > 0 THEN 1 ELSE 0 END AS has_recent_collection,

        -- ═══ Interaction Features ═══
        int_rate * dti AS int_rate_x_dti,

        CASE WHEN annual_inc > 0 AND loan_amnt > 0
             THEN int_rate * (loan_amnt / annual_inc)
             ELSE 0
        END AS rate_burden,

        -- ═══ Installment Burden ═══
        CASE WHEN loan_amnt > 0
             THEN installment * term / loan_amnt
             ELSE 0
        END AS total_payment_to_loan

    FROM clean_data
    """

    result = con.execute(feature_query).fetchdf()
    con.close()

    print(f"  Generated {len(result.columns) - len(df.columns)} new engineered features")

    # ── Target encoding for high-cardinality categoricals ────────────────
    print("\n[3/5] Target-encoding high-cardinality categoricals...")

    # Split: encode only on data before 2016 to avoid leakage
    train_mask = result["issue_year"] < 2016
    global_mean = result.loc[train_mask, "target"].mean()

    high_card_cols = []
    for col in ["sub_grade", "zip_code", "addr_state", "emp_title", "purpose"]:
        if col not in result.columns:
            continue

        # Compute smoothed target encoding on training data only
        stats = result.loc[train_mask].groupby(col)["target"].agg(["mean", "count"])
        # Bayesian smoothing: blend category mean with global mean
        smoothing = 100  # smoothing factor
        stats["smoothed"] = (
            (stats["count"] * stats["mean"] + smoothing * global_mean)
            / (stats["count"] + smoothing)
        )
        encoding_map = stats["smoothed"].to_dict()
        result[f"{col}_encoded"] = result[col].map(encoding_map).fillna(global_mean)
        high_card_cols.append(col)
        print(f"  {col}: {result[col].nunique()} categories → target-encoded")

    # Drop the original high-cardinality string columns (keep encoded versions)
    result.drop(columns=[c for c in high_card_cols if c in result.columns], inplace=True)

    # ── Categoricals for LightGBM native handling ────────────────────────
    print("\n[4/5] Preparing categorical columns...")
    low_card_cats = ["term", "grade", "home_ownership", "verification_status",
                     "application_type", "initial_list_status"]
    for col in low_card_cats:
        if col in result.columns:
            result[col] = result[col].astype("category")

    # ── Cap extreme values ───────────────────────────────────────────────
    # Cap engineered ratios at reasonable bounds to prevent extreme outliers
    ratio_cols = [
        "loan_to_income_ratio", "payment_to_income_ratio",
        "revol_bal_to_income", "revolving_to_loan_ratio",
        "avg_balance_per_account",
    ]
    for col in ratio_cols:
        if col in result.columns:
            p99 = result[col].quantile(0.99)
            result[col] = result[col].clip(upper=p99)

    # ── Save ─────────────────────────────────────────────────────────────
    print("\n[5/5] Saving engineered features...")

    # Replace any remaining infinities
    result = result.replace([np.inf, -np.inf], 0)

    result.to_parquet(output_path, index=False)

    elapsed = time.time() - start

    # ── Generate feature dictionary ──────────────────────────────────────
    feature_dict = {
        "time_features": {
            "credit_history_months": "Months between first credit line and loan issue date",
            "issue_year": "Year the loan was issued",
            "issue_month": "Month the loan was issued",
        },
        "income_affordability": {
            "loan_to_income_ratio": "Loan amount / annual income",
            "payment_to_income_ratio": "Annual installment payments / annual income",
            "revol_bal_to_income": "Revolving balance / annual income",
        },
        "credit_utilization": {
            "credit_utilization_ratio": "Revolving balance / total revolving credit limit",
            "available_credit_pct": "Available credit as % of total limit",
        },
        "account_features": {
            "open_acc_ratio": "Open accounts / total accounts",
            "delinquency_ratio": "Delinquencies / total accounts",
            "inquiry_to_open_ratio": "Recent inquiries / open accounts",
            "public_record_ratio": "Public records / total accounts",
            "avg_balance_per_account": "Average current balance per account",
        },
        "balance_features": {
            "revolving_to_loan_ratio": "Revolving balance / loan amount",
            "total_utilization": "Total current balance / total high credit limit",
        },
        "risk_indicators": {
            "currently_delinquent_flag": "1 if any accounts currently delinquent",
            "has_public_record": "1 if any public records",
            "has_bankruptcy": "1 if any bankruptcies",
            "has_recent_delinquency": "1 if delinquency in last 2 years",
            "has_recent_collection": "1 if collection in last 12 months",
        },
        "interaction_features": {
            "int_rate_x_dti": "Interest rate × debt-to-income (risk interaction)",
            "rate_burden": "Interest rate × loan-to-income ratio",
            "total_payment_to_loan": "Total payments / loan amount (cost multiplier)",
        },
        "target_encoded": {
            f"{c}_encoded": f"Smoothed target encoding of {c}"
            for c in high_card_cols
        },
    }

    os.makedirs("reports", exist_ok=True)
    with open("reports/feature_dictionary.json", "w") as f:
        json.dump(feature_dict, f, indent=2)

    # Also as markdown
    md_lines = ["# Feature Dictionary (Phase 2)\n"]
    for group, features in feature_dict.items():
        md_lines.append(f"\n## {group.replace('_', ' ').title()}\n")
        md_lines.append("| Feature | Description |")
        md_lines.append("|---------|-------------|")
        for feat, desc in features.items():
            md_lines.append(f"| `{feat}` | {desc} |")
    md_lines.append(f"\n\n**Total engineered features**: {len(result.columns) - 1} (excluding target)")
    md_lines.append(f"\n**Processing time**: {elapsed:.1f}s")

    with open("reports/feature_dictionary.md", "w") as f:
        f.write("\n".join(md_lines))

    print(f"\n{'=' * 70}")
    print(f"  FEATURE ENGINEERING COMPLETE")
    print(f"  Output: {output_path}")
    print(f"  Rows: {len(result):,}  |  Features: {len(result.columns)}")
    print(f"  Feature dictionary: reports/feature_dictionary.md")
    print(f"  Time: {elapsed:.1f}s")
    print(f"{'=' * 70}")

    return result


if __name__ == "__main__":
    engineer_features()
