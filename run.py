"""
One-click launcher for the Credit Risk Dashboard.
Automatically runs the full pipeline if model artifacts are missing,
then launches the Streamlit dashboard.

Usage:  python run.py
"""

import subprocess
import sys
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))


def check_artifacts():
    """Return True if essential model artifacts exist and are readable."""
    import pandas as pd

    required = [
        "models/lgbm_model.joblib",
        "models/feature_names.joblib",
        "models/metrics.json",
    ]
    for p in required:
        if not os.path.exists(p):
            return False
    try:
        pd.read_parquet("models/background_data.parquet")
    except Exception:
        return False
    return True


def run_pipeline():
    """Run the ML pipeline steps that are needed."""
    steps = [
        ("Phase 1: Data Cleaning",        "src/data_cleaning.py"),
        ("Phase 2: Feature Engineering",   "src/feature_engineering.py"),
        ("Phase 3: Feature Store",         "src/feature_store.py"),
        ("Phase 4: Model Training",        "src/model_training.py"),
        ("Phase 5: Explainability (SHAP)", "src/explainability.py"),
        ("Phase 7: Scenario Simulation",   "src/scenario_simulation.py"),
    ]

    # Skip data prep if processed features already exist
    if os.path.exists("data/processed/features.parquet"):
        print("[*] Processed features found — skipping Phases 1-3.\n")
        steps = [s for s in steps if "Phase 1" not in s[0]
                 and "Phase 2" not in s[0] and "Phase 3" not in s[0]]

    for label, script in steps:
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")
        result = subprocess.run(
            [sys.executable, script],
            encoding="utf-8", errors="replace",
        )
        if result.returncode != 0:
            print(f"\n[!] {label} failed (exit code {result.returncode}).")
            print("    Fix the issue above and re-run this script.")
            sys.exit(1)

    print("\n" + "=" * 60)
    print("  ALL PIPELINE STEPS COMPLETE")
    print("=" * 60)


def main():
    print("=" * 60)
    print("  Credit Risk Intelligence Platform — Launcher")
    print("=" * 60)

    if not check_artifacts():
        print("\n[*] Model artifacts missing or incompatible. Running pipeline...\n")
        run_pipeline()
    else:
        print("\n[*] All model artifacts are ready.\n")

    print("\n[*] Launching Streamlit dashboard...\n")
    subprocess.run([sys.executable, "-m", "streamlit", "run", "dashboard/app.py"])


if __name__ == "__main__":
    main()
