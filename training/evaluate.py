"""Standalone evaluation that loads persisted artifacts and prints metrics."""
from __future__ import annotations
import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.utils.preprocessing import clean_dataframe, load_raw, split_features_target  # noqa: E402
from app.utils.feature_engineering import add_engineered_features  # noqa: E402
from training.config import (FEATURES_PATH, MODEL_PATH, RANDOM_STATE,
                             RAW_CSV, SCALER_PATH, TEST_SIZE, THRESHOLD_PATH)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("evaluate")


def main() -> dict:
    df = clean_dataframe(load_raw(RAW_CSV))
    df = add_engineered_features(df)
    X, y = split_features_target(df, target="Churn")
    if "customerID" in X.columns:
        X = X.drop(columns=["customerID"])
    X_enc = pd.get_dummies(X, drop_first=False)

    feature_columns = joblib.load(FEATURES_PATH)
    X_enc = X_enc.reindex(columns=feature_columns, fill_value=0)

    scaler = joblib.load(SCALER_PATH)
    num = [c for c in ("tenure", "MonthlyCharges", "TotalCharges",
                       "services_count", "charges_per_tenure") if c in X_enc.columns]
    X_enc[num] = scaler.transform(X_enc[num])

    _, X_test, _, y_test = train_test_split(X_enc, y, test_size=TEST_SIZE,
                                            random_state=RANDOM_STATE, stratify=y)

    model = joblib.load(MODEL_PATH)
    threshold = float(json.loads(THRESHOLD_PATH.read_text())["threshold"])
    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= threshold).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "precision": float(precision_score(y_test, preds)),
        "recall": float(recall_score(y_test, preds)),
        "f1": float(f1_score(y_test, preds)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "pr_auc": float(average_precision_score(y_test, proba)),
        "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
        "threshold": threshold,
    }
    logger.info("Evaluation metrics: %s", metrics)
    return metrics


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
