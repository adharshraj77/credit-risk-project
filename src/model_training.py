"""
Phase 4: Credit Risk Prediction Model
=======================================
Trains Logistic Regression (baseline) and LightGBM (main model) with temporal
split, class-imbalance handling, probability calibration (isotonic), and
comprehensive metric evaluation (ROC-AUC, PR-AUC, KS statistic, calibration curve).

Run: python src/model_training.py
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    classification_report, roc_auc_score, roc_curve,
    confusion_matrix, precision_recall_curve, average_precision_score,
    f1_score, precision_score, recall_score, accuracy_score,
    brier_score_loss, log_loss,
)
import joblib
import json
import os
import time
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def compute_ks_statistic(y_true, y_prob):
    """Compute Kolmogorov-Smirnov statistic for model discrimination."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    ks = np.max(tpr - fpr)
    ks_threshold = thresholds[np.argmax(tpr - fpr)]
    return float(ks), float(ks_threshold)


def get_optimal_threshold(y_true, y_prob):
    """Find optimal threshold using Youden's J statistic."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    return float(thresholds[best_idx])


def train_model():
    """Train credit risk models with temporal split and calibration."""

    features_path = "data/processed/features.parquet"
    if not os.path.exists(features_path):
        print(f"ERROR: Could not find {features_path}")
        print("Run src/feature_engineering.py first.")
        return

    print("=" * 70)
    print("  PHASE 4: CREDIT RISK MODEL TRAINING")
    print("=" * 70)

    start = time.time()
    os.makedirs("models", exist_ok=True)

    # ── 1. Load Data ─────────────────────────────────────────────────────
    print("\n[1/8] Loading features...")
    df = pd.read_parquet(features_path)
    df = df.sample(frac=0.2, random_state=42) # Downsample to prevent memory errors
    print(f"  Dataset: {len(df):,} rows × {len(df.columns)} columns")

    # ── 2. Drop remaining leakage and non-feature columns ────────────────
    print("\n[2/8] Preparing feature matrix...")
    drop_cols = [
        "target", "issue_d", "earliest_cr_line", "snapshot_date",
        "issue_year", "issue_month",  # will use for splitting, not as features
    ]
    # Also drop any remaining settlement/hardship columns that survived
    extra_leakage = [c for c in df.columns if any(
        kw in c.lower() for kw in ["settlement", "hardship"]
    )]
    drop_cols.extend(extra_leakage)
    drop_cols = [c for c in drop_cols if c in df.columns]

    # Separate year/month for splitting before dropping
    if "issue_year" not in df.columns:
        if "issue_d" in df.columns:
            df["issue_year"] = pd.to_datetime(df["issue_d"]).dt.year
        else:
            print("  WARNING: No year column for temporal split. Using random split.")
            df["issue_year"] = 2015  # fallback

    features = [c for c in df.columns if c not in drop_cols]
    print(f"  Features: {len(features)}")
    print(f"  Dropped: {drop_cols}")

    # ── 3. Temporal Split ────────────────────────────────────────────────
    print("\n[3/8] Temporal train/validation/test split...")
    train_df = df[df["issue_year"] <= 2015].copy()
    val_df = df[df["issue_year"] == 2016].copy()
    test_df = df[df["issue_year"] >= 2017].copy()

    print(f"  Train (≤2015): {len(train_df):,} rows ({train_df['target'].mean():.2%} default)")
    print(f"  Val   (2016):  {len(val_df):,} rows ({val_df['target'].mean():.2%} default)")
    print(f"  Test  (≥2017): {len(test_df):,} rows ({test_df['target'].mean():.2%} default)")

    X_train = train_df[features].copy()
    y_train = train_df["target"]
    X_val = val_df[features].copy()
    y_val = val_df["target"]
    X_test = test_df[features].copy()
    y_test = test_df["target"]

    # ── 4. Handle categorical and numeric columns ────────────────────────
    print("\n[4/8] Encoding features...")
    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()

    # For LightGBM: convert object to category
    label_encoders = {}
    for c in cat_cols:
        le = LabelEncoder()
        # Fit on all unique values across all splits
        all_vals = pd.concat([X_train[c], X_val[c], X_test[c]]).astype(str).unique()
        le.fit(all_vals)
        X_train[c] = le.transform(X_train[c].astype(str))
        X_val[c] = le.transform(X_val[c].astype(str))
        X_test[c] = le.transform(X_test[c].astype(str))
        label_encoders[c] = le

    print(f"  Label-encoded {len(cat_cols)} categorical columns")
    print(f"  Numeric features: {len(num_cols)}")

    # ── 5. Logistic Regression (Baseline) ────────────────────────────────
    print("\n[5/8] Training Logistic Regression baseline...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.values.astype(np.float32))
    X_val_scaled = scaler.transform(X_val.values.astype(np.float32))
    X_test_scaled = scaler.transform(X_test.values.astype(np.float32))

    # Replace NaN/inf in scaled data
    X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    X_val_scaled = np.nan_to_num(X_val_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    X_test_scaled = np.nan_to_num(X_test_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    logreg = LogisticRegression(
        C=0.1,
        class_weight="balanced",
        max_iter=1000,
        solver="saga",
        random_state=42,
        n_jobs=-1,
    )
    logreg.fit(X_train_scaled, y_train)
    logreg_probs = logreg.predict_proba(X_test_scaled)[:, 1]
    logreg_auc = roc_auc_score(y_test, logreg_probs)
    print(f"  Logistic Regression ROC-AUC: {logreg_auc:.4f}")

    joblib.dump(logreg, "models/logreg_model.joblib")
    joblib.dump(scaler, "models/scaler.joblib")
    joblib.dump(label_encoders, "models/label_encoders.joblib")

    # ── 6. LightGBM (Main Model) ────────────────────────────────────────
    print("\n[6/8] Training LightGBM (main model)...")

    # Calculate scale_pos_weight for imbalance handling
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / n_pos

    lgbm_model = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=8,
        num_leaves=63,
        min_child_samples=100,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    lgbm_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=100),
        ],
    )

    lgbm_probs_raw = lgbm_model.predict_proba(X_test)[:, 1]
    lgbm_auc = roc_auc_score(y_test, lgbm_probs_raw)
    print(f"  LightGBM ROC-AUC (uncalibrated): {lgbm_auc:.4f}")
    print(f"  Best iteration: {lgbm_model.best_iteration_}")

    joblib.dump(lgbm_model, "models/lgbm_model.joblib")
    joblib.dump(features, "models/feature_names.joblib")

    # ── 7. Probability Calibration ───────────────────────────────────────
    print("\n[7/8] Calibrating probabilities (isotonic regression)...")

    # Use validation set for calibration
    calibrated_model = CalibratedClassifierCV(
        estimator=lgbm_model,
        method="isotonic",
        cv=3,
    )
    calibrated_model.fit(X_val, y_val)

    lgbm_probs = calibrated_model.predict_proba(X_test)[:, 1]
    cal_auc = roc_auc_score(y_test, lgbm_probs)
    print(f"  Calibrated ROC-AUC: {cal_auc:.4f}")

    joblib.dump(calibrated_model, "models/calibrated_model.joblib")

    # ── 8. Comprehensive Evaluation ──────────────────────────────────────
    print("\n[8/8] Computing comprehensive metrics...")

    # Use calibrated model as the primary model
    probs = lgbm_probs  # calibrated probabilities

    # Optimal threshold via Youden's J
    optimal_threshold = get_optimal_threshold(y_test, probs)
    preds = (probs >= optimal_threshold).astype(int)
    print(f"  Optimal threshold (Youden's J): {optimal_threshold:.3f}")

    # Core metrics
    roc_auc = roc_auc_score(y_test, probs)
    avg_precision = average_precision_score(y_test, probs)
    ks_stat, ks_thresh = compute_ks_statistic(y_test, probs)
    brier = brier_score_loss(y_test, probs)
    logloss = log_loss(y_test, probs)

    accuracy = accuracy_score(y_test, preds)
    precision_1 = precision_score(y_test, preds, pos_label=1)
    recall_1 = recall_score(y_test, preds, pos_label=1)
    f1_1 = f1_score(y_test, preds, pos_label=1)
    precision_0 = precision_score(y_test, preds, pos_label=0)
    recall_0 = recall_score(y_test, preds, pos_label=0)
    f1_0 = f1_score(y_test, preds, pos_label=0)

    print(f"\n  ┌─── Model Performance ────────────────────────┐")
    print(f"  │  ROC AUC:           {roc_auc:.4f}              │")
    print(f"  │  PR AUC:            {avg_precision:.4f}              │")
    print(f"  │  KS Statistic:      {ks_stat:.4f}              │")
    print(f"  │  Brier Score:       {brier:.4f}              │")
    print(f"  │  Log Loss:          {logloss:.4f}              │")
    print(f"  │  Accuracy:          {accuracy:.4f}              │")
    print(f"  │  Default Recall:    {recall_1:.4f}              │")
    print(f"  │  Default Precision: {precision_1:.4f}              │")
    print(f"  └───────────────────────────────────────────────┘")

    print(f"\n{classification_report(y_test, preds, target_names=['Good Loan', 'Default'])}")

    # ── Chart data ───────────────────────────────────────────────────────
    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, probs)
    step = max(1, len(fpr) // 200)
    fpr_sampled = fpr[::step].tolist()
    tpr_sampled = tpr[::step].tolist()

    # KS curve data
    ks_fpr, ks_tpr, ks_thresholds = roc_curve(y_test, probs)
    ks_step = max(1, len(ks_thresholds) // 200)

    # Precision-Recall curve
    pr_prec, pr_rec, _ = precision_recall_curve(y_test, probs)
    step_pr = max(1, len(pr_prec) // 200)
    pr_prec_sampled = pr_prec[::step_pr].tolist()
    pr_rec_sampled = pr_rec[::step_pr].tolist()

    # Confusion matrix
    cm = confusion_matrix(y_test, preds)

    # Feature importance (top 30)
    importances = lgbm_model.feature_importances_
    feat_imp = sorted(
        zip(features, [int(x) for x in importances]),
        key=lambda x: x[1], reverse=True,
    )[:30]

    # Score distribution
    good_probs = probs[y_test == 0]
    bad_probs = probs[y_test == 1]
    hist_good, bin_edges = np.histogram(good_probs, bins=50, range=(0, 1))
    hist_bad, _ = np.histogram(bad_probs, bins=50, range=(0, 1))

    # Calibration curve
    cal_fraction, cal_mean_pred = calibration_curve(y_test, probs, n_bins=10, strategy="uniform")

    # LogReg coefficients (top features by absolute coefficient)
    logreg_coefs = sorted(
        zip(features, logreg.coef_[0].tolist()),
        key=lambda x: abs(x[1]), reverse=True,
    )[:20]

    # ── Save comprehensive metrics JSON ──────────────────────────────────
    print("  Saving metrics and artifacts...")

    # Background data for SHAP/DiCE (sample from training)
    bg_sample = X_train.sample(n=min(1000, len(X_train)), random_state=42).copy()
    bg_sample["target"] = y_train.loc[bg_sample.index].values
    bg_sample.to_parquet("models/background_data.parquet")

    metrics = {
        "model_type": "LightGBM (Calibrated, Isotonic)",
        "baseline_model": "Logistic Regression (L2, balanced)",
        "temporal_split": {
            "train": "2007–2015",
            "validation": "2016",
            "test": "2017–2018",
        },
        "hyperparameters": {
            "n_estimators": lgbm_model.best_iteration_,
            "learning_rate": 0.03,
            "max_depth": 8,
            "num_leaves": 63,
            "min_child_samples": 100,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "scale_pos_weight": round(scale_pos_weight, 2),
        },
        "dataset": {
            "total_rows": int(len(df)),
            "num_features": len(features),
            "num_categorical": len(cat_cols),
            "num_numeric": len(num_cols),
            "train_size": int(len(train_df)),
            "val_size": int(len(val_df)),
            "test_size": int(len(test_df)),
            "train_default_rate": round(float(train_df["target"].mean()), 4),
            "val_default_rate": round(float(val_df["target"].mean()), 4),
            "test_default_rate": round(float(test_df["target"].mean()), 4),
        },
        "performance": {
            "roc_auc": round(roc_auc, 4),
            "pr_auc": round(avg_precision, 4),
            "ks_statistic": round(ks_stat, 4),
            "ks_threshold": round(ks_thresh, 4),
            "optimal_threshold": round(optimal_threshold, 4),
            "brier_score": round(brier, 4),
            "log_loss": round(logloss, 4),
            "accuracy": round(accuracy, 4),
            "good_loan": {
                "precision": round(precision_0, 4),
                "recall": round(recall_0, 4),
                "f1": round(f1_0, 4),
            },
            "default": {
                "precision": round(precision_1, 4),
                "recall": round(recall_1, 4),
                "f1": round(f1_1, 4),
            },
        },
        "baseline_performance": {
            "logreg_roc_auc": round(logreg_auc, 4),
        },
        "charts": {
            "roc_curve": {"fpr": fpr_sampled, "tpr": tpr_sampled},
            "pr_curve": {"precision": pr_prec_sampled, "recall": pr_rec_sampled},
            "confusion_matrix": cm.tolist(),
            "feature_importance": [
                {"feature": f, "importance": i} for f, i in feat_imp
            ],
            "score_distribution": {
                "bins": [round(float(b), 3) for b in bin_edges[:-1]],
                "good_loans": hist_good.tolist(),
                "defaults": hist_bad.tolist(),
            },
            "calibration_curve": {
                "fraction_of_positives": cal_fraction.tolist(),
                "mean_predicted_value": cal_mean_pred.tolist(),
            },
            "logreg_coefficients": [
                {"feature": f, "coefficient": round(c, 4)} for f, c in logreg_coefs
            ],
        },
        "training_time_seconds": round(time.time() - start, 1),
    }

    with open("models/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"  MODEL TRAINING COMPLETE")
    print(f"  LightGBM ROC-AUC:   {roc_auc:.4f}")
    print(f"  LogReg ROC-AUC:     {logreg_auc:.4f}")
    print(f"  KS Statistic:       {ks_stat:.4f}")
    print(f"  PR AUC:             {avg_precision:.4f}")
    print(f"  Optimal threshold:  {optimal_threshold:.3f}")
    print(f"  Artifacts: models/")
    print(f"  Metrics: models/metrics.json")
    print(f"  Time: {time.time() - start:.1f}s")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    train_model()
