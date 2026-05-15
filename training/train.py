"""End-to-end training script for the churn model."""
from __future__ import annotations
import json
import logging
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.utils.preprocessing import clean_dataframe, load_raw, split_features_target  # noqa: E402
from app.utils.feature_engineering import add_engineered_features  # noqa: E402
from training.config import (FEATURES_PATH, MODEL_PATH, MODELS_DIR,
                             METRICS_PATH, RANDOM_STATE, RAW_CSV,
                             SCALER_PATH, TEST_SIZE, THRESHOLD_PATH,
                             TRAIN_SAMPLE_PATH, XGB_PARAM_GRID_FAST)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("training")

NUM_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges",
                "services_count", "charges_per_tenure"]


def build_feature_matrix(df: pd.DataFrame):
    df = clean_dataframe(df)
    df = add_engineered_features(df)
    X, y = split_features_target(df, target="Churn")
    if "customerID" in X.columns:
        X = X.drop(columns=["customerID"])
    X_enc = pd.get_dummies(X, drop_first=False)
    return X_enc, y


def youden_threshold(y_true, y_proba) -> float:
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    j = tpr - fpr
    idx = int(np.argmax(j))
    return float(thresholds[idx])


def main(use_full_grid: bool = False) -> dict:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading dataset")
    df = load_raw(RAW_CSV)
    X, y = build_feature_matrix(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    num_present = [c for c in NUM_FEATURES if c in X_train.columns]
    X_train[num_present] = scaler.fit_transform(X_train[num_present])
    X_test[num_present] = scaler.transform(X_test[num_present])

    logger.info("Training Logistic Regression baseline")
    baseline = LogisticRegression(max_iter=1000, class_weight="balanced",
                                  random_state=RANDOM_STATE)
    baseline.fit(X_train, y_train)
    base_proba = baseline.predict_proba(X_test)[:, 1]
    baseline_auc = roc_auc_score(y_test, base_proba)
    logger.info("Baseline ROC-AUC: %.4f", baseline_auc)

    pos = int(y_train.sum())
    neg = int(len(y_train) - pos)
    spw = max(neg / max(pos, 1), 1.0)

    xgb = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        scale_pos_weight=spw,
        tree_method="hist",
        n_jobs=-1,
    )

    grid = XGB_PARAM_GRID_FAST  # fast grid keeps training under a minute
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    logger.info("Running GridSearchCV with grid=%s", grid)
    search = GridSearchCV(xgb, grid, cv=cv, scoring="roc_auc", n_jobs=-1, verbose=0)
    search.fit(X_train, y_train)
    model = search.best_estimator_
    logger.info("Best params: %s", search.best_params_)

    proba = model.predict_proba(X_test)[:, 1]
    threshold = youden_threshold(y_test, proba)
    preds = (proba >= threshold).astype(int)

    metrics = {
        "baseline_logreg_roc_auc": float(baseline_auc),
        "best_params": search.best_params_,
        "accuracy": float(accuracy_score(y_test, preds)),
        "precision": float(precision_score(y_test, preds)),
        "recall": float(recall_score(y_test, preds)),
        "f1": float(f1_score(y_test, preds)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "pr_auc": float(average_precision_score(y_test, proba)),
        "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
        "optimal_threshold_youden_j": float(threshold),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }
    logger.info("Final metrics: %s", {k: v for k, v in metrics.items() if k != "best_params"})

    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(list(X_train.columns), FEATURES_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    THRESHOLD_PATH.write_text(json.dumps({"threshold": threshold}, indent=2))

    # Save a small reference sample (untransformed numeric cols) for drift detection
    ref = df.sample(min(1000, len(df)), random_state=RANDOM_STATE)
    ref.to_parquet(TRAIN_SAMPLE_PATH, index=False)

    logger.info("Artifacts written to %s", MODELS_DIR)
    return metrics


if __name__ == "__main__":
    main()
