"""
Phase 3: Time-Versioned Feature Store
=======================================
Builds a DuckDB-based feature store with monthly snapshots of the loan portfolio.
Provides API functions for retrieving current vectors, historical timelines,
and computing Population Stability Index (PSI) drift between monthly snapshots.

Run: python src/feature_store.py
"""

import duckdb
import pandas as pd
import numpy as np
import os
import time
import json


def compute_psi(expected, actual, bins=10):
    """
    Compute Population Stability Index between two distributions.
    PSI < 0.1: no significant change
    PSI 0.1–0.2: moderate change
    PSI > 0.2: significant change
    """
    # Create bins from expected distribution
    breakpoints = np.linspace(0, 1, bins + 1)

    # Use percentiles from expected for better binning
    try:
        breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))
        breakpoints = np.unique(breakpoints)
        if len(breakpoints) < 3:
            return 0.0
    except Exception:
        return 0.0

    expected_counts = np.histogram(expected, bins=breakpoints)[0]
    actual_counts = np.histogram(actual, bins=breakpoints)[0]

    # Normalize to proportions, add small epsilon to avoid division by zero
    eps = 1e-6
    expected_pct = expected_counts / (expected_counts.sum() + eps) + eps
    actual_pct = actual_counts / (actual_counts.sum() + eps) + eps

    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi)


def build_feature_store():
    """Build time-versioned feature store with monthly snapshots and drift metrics."""

    features_path = "data/processed/features.parquet"
    store_path = "data/feature_store.duckdb"
    snapshots_dir = "data/snapshots"

    if not os.path.exists(features_path):
        print(f"ERROR: Could not find {features_path}")
        print("Run src/feature_engineering.py first.")
        return

    print("=" * 70)
    print("  PHASE 3: TIME-VERSIONED FEATURE STORE")
    print("=" * 70)

    start = time.time()
    os.makedirs(snapshots_dir, exist_ok=True)

    # ── Load features ────────────────────────────────────────────────────
    print("\n[1/4] Loading engineered features...")
    df = pd.read_parquet(features_path)
    print(f"  {len(df):,} rows × {len(df.columns)} columns")

    # ── Create snapshot_date column ──────────────────────────────────────
    print("\n[2/4] Building monthly snapshots...")

    # Use issue_year and issue_month to create snapshot dates
    if "issue_year" in df.columns and "issue_month" in df.columns:
        df["snapshot_date"] = pd.to_datetime(
            df["issue_year"].astype(int).astype(str) + "-"
            + df["issue_month"].astype(int).astype(str).str.zfill(2) + "-01"
        )
    elif "issue_d" in df.columns:
        df["snapshot_date"] = df["issue_d"].dt.to_period("M").dt.to_timestamp()
    else:
        print("  WARNING: No date column found. Using synthetic monthly snapshots.")
        df["snapshot_date"] = pd.Timestamp("2015-01-01")

    # Get unique months sorted
    months = sorted(df["snapshot_date"].dropna().unique())
    print(f"  Found {len(months)} unique monthly snapshots")
    if len(months) > 0:
        print(f"  Range: {pd.Timestamp(months[0]).strftime('%Y-%m')} → "
              f"{pd.Timestamp(months[-1]).strftime('%Y-%m')}")

    # ── Build DuckDB store ───────────────────────────────────────────────
    print("\n[3/4] Building DuckDB feature store...")

    # Remove old store if exists
    if os.path.exists(store_path):
        os.remove(store_path)

    con = duckdb.connect(database=store_path)

    # Create main features table
    con.execute("CREATE TABLE features AS SELECT * FROM df")

    # Create an index on snapshot_date for fast lookups
    con.execute("""
        CREATE INDEX idx_snapshot ON features (snapshot_date)
    """)

    # Get row counts per snapshot
    snapshot_counts = con.execute("""
        SELECT snapshot_date, COUNT(*) as n_loans, AVG(target) as default_rate
        FROM features
        GROUP BY snapshot_date
        ORDER BY snapshot_date
    """).fetchdf()

    print(f"  Stored {len(df):,} rows in DuckDB")
    print(f"  Snapshots with data: {len(snapshot_counts)}")

    # ── Compute PSI drift between consecutive months ─────────────────────
    print("\n[4/4] Computing Population Stability Index (PSI) drift...")

    # We compute PSI on the model's predicted probability proxy (int_rate as proxy)
    # and on key features between consecutive months
    drift_features = ["int_rate", "dti", "annual_inc", "loan_amnt", "revol_util"]
    drift_features = [f for f in drift_features if f in df.columns]

    drift_results = []
    prev_month_data = None

    for i, month in enumerate(months):
        month_data = df[df["snapshot_date"] == month]

        if prev_month_data is not None and len(month_data) > 50:
            month_drift = {
                "snapshot_date": pd.Timestamp(month).strftime("%Y-%m"),
                "n_loans": int(len(month_data)),
                "default_rate": round(float(month_data["target"].mean()), 4),
            }

            for feat in drift_features:
                if feat in month_data.columns and feat in prev_month_data.columns:
                    expected = prev_month_data[feat].dropna().values
                    actual = month_data[feat].dropna().values
                    if len(expected) > 10 and len(actual) > 10:
                        psi = compute_psi(expected, actual, bins=10)
                        month_drift[f"psi_{feat}"] = round(psi, 6)

            # Compute overall PSI as average across features
            psi_values = [v for k, v in month_drift.items() if k.startswith("psi_")]
            month_drift["psi_overall"] = round(np.mean(psi_values), 6) if psi_values else 0.0
            drift_results.append(month_drift)

        prev_month_data = month_data

    drift_df = pd.DataFrame(drift_results)

    # Save drift metrics
    if len(drift_df) > 0:
        con.execute("CREATE TABLE drift_metrics AS SELECT * FROM drift_df")
        drift_df.to_csv("reports/drift_metrics.csv", index=False)
        print(f"  Computed PSI for {len(drift_df)} month-pairs")

        # Flag significant drift
        sig_drift = drift_df[drift_df["psi_overall"] > 0.2]
        if len(sig_drift) > 0:
            print(f"  ⚠ {len(sig_drift)} months with significant drift (PSI > 0.2)")
        else:
            print(f"  ✓ No significant drift detected (all PSI < 0.2)")
    else:
        print("  Skipped — not enough monthly snapshots for drift computation")

    con.close()

    elapsed = time.time() - start

    # Save store metadata
    store_meta = {
        "store_path": store_path,
        "total_rows": int(len(df)),
        "n_snapshots": int(len(months)),
        "date_range": {
            "start": pd.Timestamp(months[0]).strftime("%Y-%m") if months else None,
            "end": pd.Timestamp(months[-1]).strftime("%Y-%m") if months else None,
        },
        "drift_features": drift_features,
        "n_drift_alerts": int(len(drift_df[drift_df["psi_overall"] > 0.2])) if len(drift_df) > 0 else 0,
        "build_time_seconds": round(elapsed, 1),
    }
    with open("reports/feature_store_metadata.json", "w") as f:
        json.dump(store_meta, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"  FEATURE STORE COMPLETE")
    print(f"  DuckDB: {store_path}")
    print(f"  Snapshots: {len(months)} months")
    print(f"  Drift report: reports/drift_metrics.csv")
    print(f"  Time: {elapsed:.1f}s")
    print(f"{'=' * 70}")


# ── API Functions ────────────────────────────────────────────────────────

def get_current_vector(store_path="data/feature_store.duckdb", loan_index=0):
    """Get the latest feature vector for a given loan (by row index)."""
    con = duckdb.connect(database=store_path, read_only=True)
    result = con.execute(f"SELECT * FROM features LIMIT 1 OFFSET {loan_index}").fetchdf()
    con.close()
    return result


def get_history(store_path="data/feature_store.duckdb", year_month=None):
    """Get all loans for a given snapshot month (YYYY-MM format)."""
    con = duckdb.connect(database=store_path, read_only=True)
    if year_month:
        result = con.execute(
            "SELECT * FROM features WHERE strftime(snapshot_date, '%Y-%m') = ?",
            [year_month]
        ).fetchdf()
    else:
        result = con.execute("SELECT * FROM features").fetchdf()
    con.close()
    return result


def get_snapshot_summary(store_path="data/feature_store.duckdb"):
    """Get summary statistics per monthly snapshot."""
    if not os.path.exists(store_path):
        fallback_csv = "reports/snapshot_summary.csv"
        if os.path.exists(fallback_csv):
            return pd.read_csv(fallback_csv, parse_dates=["snapshot_date"])
        return pd.DataFrame()
        
    con = duckdb.connect(database=store_path, read_only=True)
    result = con.execute("""
        SELECT
            snapshot_date,
            COUNT(*) as n_loans,
            AVG(target) as default_rate,
            AVG(loan_amnt) as avg_loan_amnt,
            AVG(int_rate) as avg_int_rate,
            AVG(dti) as avg_dti
        FROM features
        GROUP BY snapshot_date
        ORDER BY snapshot_date
    """).fetchdf()
    con.close()
    return result


def get_drift_metrics(store_path="data/feature_store.duckdb"):
    """Get PSI drift metrics from the feature store."""
    if not os.path.exists(store_path):
        fallback_csv = "reports/drift_metrics.csv"
        if os.path.exists(fallback_csv):
            return pd.read_csv(fallback_csv)
        return pd.DataFrame()

    con = duckdb.connect(database=store_path, read_only=True)
    try:
        result = con.execute("SELECT * FROM drift_metrics ORDER BY snapshot_date").fetchdf()
    except Exception:
        result = pd.DataFrame()
    con.close()
    return result


if __name__ == "__main__":
    build_feature_store()
