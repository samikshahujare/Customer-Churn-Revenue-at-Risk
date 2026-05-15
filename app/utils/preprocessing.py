"""Data cleaning and preprocessing utilities for the churn pipeline."""
from __future__ import annotations
import logging
from typing import Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def load_raw(csv_path) -> pd.DataFrame:
    """Load the raw Telco churn CSV."""
    logger.info("Loading raw CSV from %s", csv_path)
    df = pd.read_csv(csv_path)
    return df


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply standard cleaning to the Telco churn dataset.

    - Convert TotalCharges to numeric (blank strings -> NaN -> filled with median).
    - Strip whitespace from object columns.
    - Normalize SeniorCitizen to Yes/No string for one-hot consistency.
    """
    df = df.copy()
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(
            df["TotalCharges"].astype(str).str.strip().replace("", np.nan),
            errors="coerce",
        )
        median_total = df["TotalCharges"].median()
        df["TotalCharges"] = df["TotalCharges"].fillna(median_total)

    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()

    if "SeniorCitizen" in df.columns and df["SeniorCitizen"].dtype != object:
        df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"}).fillna("No")

    return df


def split_features_target(df: pd.DataFrame, target: str = "Churn"):
    """Split off the target column and binarize Yes/No to 1/0."""
    y = df[target].map({"Yes": 1, "No": 0}).astype(int)
    X = df.drop(columns=[target])
    return X, y


REQUIRED_COLUMNS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges",
]


def validate_schema(df: pd.DataFrame) -> Optional[str]:
    """Return None if schema OK, else a human-readable error string."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        return f"Missing required columns: {', '.join(missing)}"
    return None
