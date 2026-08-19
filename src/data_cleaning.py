"""
Phase 1: Data Cleaning Pipeline
=================================
Loads raw Lending Club data, removes leakage columns, handles missing values
with informative-missingness flags, defines the target variable, fixes data types,
and saves a clean Parquet file.

Run: python src/data_cleaning.py
"""

import pandas as pd
import numpy as np
import os
import time
import json


def clean_lending_club_data():
    """Clean Lending Club loan data for credit risk modeling."""

    # Try multiple possible locations for the raw data
    candidates = [
        "data/raw/accepted_2007_to_2018Q4.csv",
        "data/raw/accepted_2007_to_2018q4.csv/accepted_2007_to_2018Q4.csv",
        "data/raw/accepted_2007_to_2018Q4.csv.gz",
    ]
    raw_data_path = None
    for path in candidates:
        if os.path.isfile(path):
            raw_data_path = path
            break
    if raw_data_path is None:
        print("ERROR: Could not find raw Lending Club data in data/raw/")
        return None

    print("=" * 70)
    print("  PHASE 1: DATA CLEANING PIPELINE")
    print("=" * 70)

    start = time.time()

    # ── 1. Load ──────────────────────────────────────────────────────────
    print("\n[1/8] Loading raw data...")
    df = pd.read_csv(raw_data_path, low_memory=False)
    print(f"  Loaded {len(df):,} rows × {len(df.columns)} columns")

    # ── 2. Target definition ─────────────────────────────────────────────
    print("\n[2/8] Defining target variable...")
    valid_statuses = ["Charged Off", "Default", "Fully Paid"]
    before = len(df)
    df = df[df["loan_status"].isin(valid_statuses)].copy()
    dropped_intermediate = before - len(df)
    print(f"  Kept {len(df):,} rows with known outcomes")
    print(f"  Dropped {dropped_intermediate:,} rows (Current, Grace Period, Late, etc.)")

    df["target"] = (df["loan_status"].isin(["Charged Off", "Default"])).astype(int)
    print(f"  Default rate: {df['target'].mean():.2%}")

    # ── 3. Drop leakage columns ─────────────────────────────────────────
    print("\n[3/8] Removing leakage & ID columns...")
    leakage_cols = [
        # Payment/recovery info — only known AFTER the loan outcome
        "total_rec_prncp", "total_pymnt", "total_pymnt_inv", "recoveries",
        "last_pymnt_d", "last_pymnt_amnt", "collection_recovery_fee",
        "out_prncp", "out_prncp_inv", "next_pymnt_d", "total_rec_int",
        "total_rec_late_fee", "last_credit_pull_d",
        # Settlement/hardship — only exist for defaulted loans
        "settlement_status", "settlement_date", "settlement_amount",
        "settlement_percentage", "settlement_term",
        "debt_settlement_flag", "debt_settlement_flag_date",
        "hardship_flag", "hardship_type", "hardship_reason",
        "hardship_status", "hardship_start_date", "hardship_end_date",
        "hardship_dpd", "hardship_loan_status", "hardship_amount",
        "hardship_length", "hardship_payoff_balance_amount",
        "hardship_last_payment_amount",
        "payment_plan_start_date", "pymnt_plan",
        # ID columns — no predictive value
        "id", "member_id", "url", "desc", "title",
        # funded_amnt is near-duplicate of loan_amnt set after approval
        "funded_amnt", "funded_amnt_inv",
        # Target source column
        "loan_status",
        # Policy code is constant (always 1)
        "policy_code",
    ]
    cols_to_drop = [c for c in leakage_cols if c in df.columns]
    df.drop(columns=cols_to_drop, inplace=True)
    print(f"  Dropped {len(cols_to_drop)} columns")

    # ── 4. Parse dates ───────────────────────────────────────────────────
    print("\n[4/8] Parsing date columns...")
    for c in ["issue_d", "earliest_cr_line"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], format="%b-%Y", errors="coerce")
            valid_dates = df[c].notna().sum()
            print(f"  {c}: {valid_dates:,} valid dates parsed")

    # ── 5. Fix data types ────────────────────────────────────────────────
    print("\n[5/8] Fixing data types...")

    # int_rate: "13.56%" → 13.56
    if "int_rate" in df.columns and df["int_rate"].dtype == object:
        df["int_rate"] = pd.to_numeric(
            df["int_rate"].astype(str).str.replace("%", "", regex=False),
            errors="coerce",
        )
        print("  int_rate: string → float (removed %)")

    # revol_util: "65.3%" → 65.3
    if "revol_util" in df.columns and df["revol_util"].dtype == object:
        df["revol_util"] = pd.to_numeric(
            df["revol_util"].astype(str).str.replace("%", "", regex=False),
            errors="coerce",
        )
        print("  revol_util: string → float (removed %)")

    # term: " 36 months" → 36
    if "term" in df.columns and df["term"].dtype == object:
        df["term"] = pd.to_numeric(
            df["term"].astype(str).str.extract(r"(\d+)", expand=False),
            errors="coerce",
        )
        print("  term: string → numeric")

    # emp_length: "10+ years" → 10, "< 1 year" → 0
    if "emp_length" in df.columns and df["emp_length"].dtype == object:
        emp_map = {
            "< 1 year": 0, "1 year": 1, "2 years": 2, "3 years": 3,
            "4 years": 4, "5 years": 5, "6 years": 6, "7 years": 7,
            "8 years": 8, "9 years": 9, "10+ years": 10,
        }
        df["emp_length"] = df["emp_length"].map(emp_map)
        print("  emp_length: string → numeric (0-10)")

    # zip_code: keep only first 3 digits (anonymized anyway)
    if "zip_code" in df.columns:
        df["zip_code"] = df["zip_code"].astype(str).str[:3]

    # ── 6. Informative missingness flags ─────────────────────────────────
    print("\n[6/8] Creating informative missingness flags...")
    informative_cols = [
        "mths_since_last_delinq",           # never delinquent
        "mths_since_last_record",            # no public records
        "mths_since_recent_bc",              # no recent bankcard
        "mths_since_recent_inq",             # no recent inquiry
        "mths_since_last_major_derog",       # no major derogatory
        "mths_since_recent_bc_dlq",          # no recent bankcard delinquency
        "mths_since_recent_revol_delinq",    # no recent revolving delinquency
        "il_util",                           # no installment utilization
        "emp_length",                        # employment not reported
    ]
    flags_created = 0
    for c in informative_cols:
        if c in df.columns and df[c].isnull().any():
            df[f"{c}_was_missing"] = df[c].isnull().astype(int)
            flags_created += 1
    print(f"  Created {flags_created} _was_missing indicator flags")

    # ── 7. Impute remaining missing values ───────────────────────────────
    print("\n[7/8] Imputing remaining missing values...")

    # Numeric: median imputation
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    n_num = 0
    for c in numeric_cols:
        if df[c].isnull().any():
            df[c] = df[c].fillna(df[c].median())
            n_num += 1

    # Categorical: mode imputation
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    n_cat = 0
    for c in cat_cols:
        if df[c].isnull().any():
            mode_val = df[c].mode()
            fill_val = mode_val.iloc[0] if len(mode_val) > 0 else "Unknown"
            df[c] = df[c].fillna(fill_val)
            n_cat += 1

    print(f"  Imputed {n_num} numeric columns (median)")
    print(f"  Imputed {n_cat} categorical columns (mode)")

    # Drop any all-null or single-value columns
    before_cols = len(df.columns)
    df = df.loc[:, df.nunique() > 1]
    dropped_const = before_cols - len(df.columns)
    if dropped_const:
        print(f"  Dropped {dropped_const} constant/empty columns")

    # ── 8. Save ──────────────────────────────────────────────────────────
    print("\n[8/8] Saving cleaned data...")
    os.makedirs("data/interim", exist_ok=True)
    df.to_parquet("data/interim/lending_club_clean.parquet", index=False)

    elapsed = time.time() - start

    # Save cleaning metadata
    cleaning_meta = {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "default_rate": round(float(df["target"].mean()), 4),
        "dropped_intermediate_rows": int(dropped_intermediate),
        "dropped_leakage_columns": cols_to_drop,
        "informative_missingness_flags": flags_created,
        "numeric_imputed": n_num,
        "categorical_imputed": n_cat,
        "cleaning_time_seconds": round(elapsed, 1),
    }
    os.makedirs("reports", exist_ok=True)
    with open("reports/cleaning_metadata.json", "w") as f:
        json.dump(cleaning_meta, f, indent=2)

    # Save detailed notes
    notes = f"""# Data Cleaning Notes (Phase 1)

## Summary
- **Input**: {before:,} raw rows → **Output**: {len(df):,} cleaned rows
- **Columns**: {len(df.columns)} features retained
- **Default rate**: {df['target'].mean():.2%}
- **Cleaning time**: {elapsed:.1f}s

## Target Definition
- **1 (Bad)**: Charged Off + Default
- **0 (Good)**: Fully Paid
- Dropped {dropped_intermediate:,} rows with intermediate statuses (Current, In Grace Period, Late, etc.) because their final outcome is unknown.

## Leakage Columns Dropped ({len(cols_to_drop)})
Post-outcome columns that would not be available at prediction time:
{chr(10).join(f'- `{c}`' for c in cols_to_drop)}

## Informative Missingness
Created `_was_missing` flags for {flags_created} columns where NULL carries predictive signal:
- `mths_since_last_delinq` → missing means "never delinquent"
- `mths_since_last_record` → missing means "no public records"
- `emp_length` → missing means "employment not reported"
- etc.

## Data Type Fixes
- `int_rate`: Stripped "%" suffix, converted to float
- `revol_util`: Stripped "%" suffix, converted to float
- `term`: Extracted numeric (36/60), converted to int
- `emp_length`: Mapped text ("10+ years") to numeric (0–10)
- `zip_code`: Kept first 3 digits only

## Imputation
- **Numeric ({n_num} cols)**: Median imputation
- **Categorical ({n_cat} cols)**: Mode imputation
"""
    with open("reports/data_cleaning_notes.md", "w") as f:
        f.write(notes)

    print(f"\n{'=' * 70}")
    print(f"  CLEANING COMPLETE")
    print(f"  Output: data/interim/lending_club_clean.parquet")
    print(f"  Rows: {len(df):,}  |  Columns: {len(df.columns)}")
    print(f"  Default rate: {df['target'].mean():.2%}")
    print(f"  Time: {elapsed:.1f}s")
    print(f"{'=' * 70}")

    return df


if __name__ == "__main__":
    clean_lending_club_data()
