"""Business / financial metrics derived from churn predictions."""
from __future__ import annotations
import pandas as pd
import numpy as np


def revenue_at_risk(df: pd.DataFrame) -> float:
    """Total annual revenue at risk = sum(P(churn) * MonthlyCharges * 12)."""
    if "revenue_at_risk_annual" in df.columns:
        return float(df["revenue_at_risk_annual"].sum())
    return float((df["churn_probability"] * df["MonthlyCharges"] * 12).sum())


def kpi_summary(df: pd.DataFrame, retention_uplift: float = 0.30) -> dict:
    """Return a dict of high-level KPIs for the dashboard."""
    total_customers = len(df)
    churn_rate = float(df["churn_prediction"].mean()) if "churn_prediction" in df.columns else float(df.get("Churn_binary", pd.Series([0])).mean())
    avg_monthly = float(df["MonthlyCharges"].mean())
    rar = revenue_at_risk(df)
    high_risk = int((df["churn_probability"] >= 0.5).sum()) if "churn_probability" in df.columns else 0
    expected_savings = rar * retention_uplift
    clv = avg_monthly * 24  # simple 2-year LTV approximation
    return {
        "total_customers": total_customers,
        "churn_rate": churn_rate,
        "avg_monthly_revenue": avg_monthly,
        "revenue_at_risk": rar,
        "high_risk_customers": high_risk,
        "expected_retention_savings": expected_savings,
        "approx_clv": clv,
    }


def retention_recommendations(row: pd.Series) -> list[str]:
    """Rule-based retention plays based on a customer's profile."""
    recs: list[str] = []
    contract = str(row.get("Contract", ""))
    if contract == "Month-to-month":
        recs.append("Offer 1- or 2-year contract with a loyalty discount")
    if str(row.get("TechSupport", "")) == "No":
        recs.append("Bundle premium tech support free for 6 months")
    if str(row.get("OnlineSecurity", "")) == "No":
        recs.append("Add complimentary online security package")
    if str(row.get("PaymentMethod", "")) == "Electronic check":
        recs.append("Migrate to auto-pay with a one-time bill credit")
    monthly = float(row.get("MonthlyCharges", 0) or 0)
    if monthly > 90:
        recs.append("Offer personalized retention package with usage-based discount")
    if not recs:
        recs.append("Maintain proactive engagement and quarterly satisfaction check-ins")
    return recs


def population_stability_index(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Compute PSI between a reference and current numeric distribution."""
    eps = 1e-6
    quantiles = np.linspace(0, 1, bins + 1)
    cuts = np.unique(np.quantile(reference, quantiles))
    if len(cuts) < 3:
        return 0.0
    ref_counts, _ = np.histogram(reference, bins=cuts)
    cur_counts, _ = np.histogram(current, bins=cuts)
    ref_pct = ref_counts / max(ref_counts.sum(), 1) + eps
    cur_pct = cur_counts / max(cur_counts.sum(), 1) + eps
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
