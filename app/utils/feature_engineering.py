"""Reusable feature engineering for the churn pipeline."""
from __future__ import annotations
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SERVICE_COLS = [
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies",
]
AUTO_PAYMENTS = {"Bank transfer (automatic)", "Credit card (automatic)"}


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features used by the model.

    Produces:
        tenure_bucket, services_count, charges_per_tenure,
        avg_monthly_spend_category, auto_payment_flag
    """
    df = df.copy()

    df["tenure_bucket"] = pd.cut(
        df["tenure"],
        bins=[-0.1, 6, 12, 24, 48, 72, np.inf],
        labels=["0-6m", "6-12m", "1-2y", "2-4y", "4-6y", "6y+"],
    ).astype(str)

    def _has_service(v: object) -> int:
        s = str(v)
        if s in {"No", "No phone service", "No internet service", "nan"}:
            return 0
        return 1

    df["services_count"] = sum(df[c].apply(_has_service) for c in SERVICE_COLS if c in df.columns)

    safe_tenure = df["tenure"].replace(0, 1)
    df["charges_per_tenure"] = df["TotalCharges"] / safe_tenure

    df["avg_monthly_spend_category"] = pd.cut(
        df["MonthlyCharges"],
        bins=[-0.1, 35, 65, 90, np.inf],
        labels=["Low", "Mid", "High", "Premium"],
    ).astype(str)

    df["auto_payment_flag"] = df["PaymentMethod"].isin(AUTO_PAYMENTS).astype(int)

    return df
