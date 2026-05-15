"""Prediction helpers: load artifacts and produce churn scores."""
from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Tuple
import joblib
import numpy as np
import pandas as pd

from .preprocessing import clean_dataframe
from .feature_engineering import add_engineered_features

logger = logging.getLogger(__name__)


def load_artifacts(models_dir: Path):
    """Load model, scaler, feature columns, and threshold from disk."""
    model = joblib.load(models_dir / "churn_model.pkl")
    scaler = joblib.load(models_dir / "scaler.pkl")
    feature_columns = joblib.load(models_dir / "feature_columns.pkl")
    threshold_path = models_dir / "threshold.json"
    threshold = 0.5
    if threshold_path.exists():
        threshold = float(json.loads(threshold_path.read_text())["threshold"])
    return model, scaler, feature_columns, threshold


def prepare_features(df: pd.DataFrame, feature_columns: list[str], scaler) -> pd.DataFrame:
    """Take a raw-ish dataframe and return a model-ready feature matrix."""
    df = clean_dataframe(df)
    df = add_engineered_features(df)

    drop_cols = [c for c in ("customerID", "Churn") if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    encoded = pd.get_dummies(df, drop_first=False)
    encoded = encoded.reindex(columns=feature_columns, fill_value=0)

    numeric_cols = [c for c in ("tenure", "MonthlyCharges", "TotalCharges",
                                "services_count", "charges_per_tenure")
                    if c in encoded.columns]
    encoded[numeric_cols] = scaler.transform(encoded[numeric_cols])
    return encoded


def predict_proba(model, X: pd.DataFrame) -> np.ndarray:
    """Return churn probability (positive class) array."""
    return model.predict_proba(X)[:, 1]


def categorize_risk(prob: float) -> str:
    """Map a probability to a Low / Medium / High / Critical risk label."""
    if prob >= 0.75:
        return "Critical"
    if prob >= 0.5:
        return "High"
    if prob >= 0.25:
        return "Medium"
    return "Low"


def add_predictions(df_raw: pd.DataFrame, model, scaler, feature_columns, threshold: float) -> pd.DataFrame:
    """Run the full pipeline on a raw dataframe and append prediction columns."""
    X = prepare_features(df_raw, feature_columns, scaler)
    proba = predict_proba(model, X)
    out = df_raw.copy()
    out["churn_probability"] = np.round(proba, 4)
    out["churn_prediction"] = (proba >= threshold).astype(int)
    out["risk_category"] = [categorize_risk(p) for p in proba]
    if "MonthlyCharges" in out.columns:
        out["revenue_at_risk_annual"] = np.round(out["churn_probability"] * out["MonthlyCharges"] * 12, 2)
    return out
