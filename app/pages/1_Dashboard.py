"""Executive analytics dashboard."""
from __future__ import annotations
import logging
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app.utils.preprocessing import clean_dataframe, load_raw  # noqa: E402
from app.utils.feature_engineering import add_engineered_features  # noqa: E402
from app.utils.prediction import add_predictions, load_artifacts  # noqa: E402
from app.utils.business_metrics import kpi_summary  # noqa: E402
from app.utils.visualizations import (churn_by_category, churn_by_tenure_bucket,
                                      churn_distribution, correlation_heatmap,
                                      monthly_charges_vs_churn,
                                      revenue_at_risk_segmentation,
                                      top_churn_drivers)
from training.config import MODELS_DIR, RAW_CSV  # noqa: E402

logger = logging.getLogger("dashboard")
st.set_page_config(page_title="Dashboard", layout="wide", page_icon="📊")
st.title("Executive Dashboard")


@st.cache_resource(show_spinner=False)
def _artifacts():
    return load_artifacts(MODELS_DIR)


@st.cache_data(show_spinner=True)
def _scored_dataset() -> pd.DataFrame:
    df = clean_dataframe(load_raw(RAW_CSV))
    df = add_engineered_features(df)
    model, scaler, feat, thr = _artifacts()
    return add_predictions(df, model, scaler, feat, thr)


try:
    df = _scored_dataset()
except Exception as exc:  # noqa: BLE001
    logger.exception("Failed to load scored dataset")
    st.error(f"Could not load dataset or model artifacts: {exc}")
    st.stop()

# ---------- Filters ----------
with st.sidebar:
    st.header("Filters")
    contracts = st.multiselect("Contract", sorted(df["Contract"].unique()),
                               default=list(sorted(df["Contract"].unique())))
    internet = st.multiselect("Internet Service", sorted(df["InternetService"].unique()),
                              default=list(sorted(df["InternetService"].unique())))
    payment = st.multiselect("Payment Method", sorted(df["PaymentMethod"].unique()),
                             default=list(sorted(df["PaymentMethod"].unique())))
    tenure_min, tenure_max = st.slider("Tenure (months)", 0, int(df["tenure"].max()),
                                       (0, int(df["tenure"].max())))

mask = (
    df["Contract"].isin(contracts)
    & df["InternetService"].isin(internet)
    & df["PaymentMethod"].isin(payment)
    & df["tenure"].between(tenure_min, tenure_max)
)
fdf = df[mask].copy()
st.caption(f"Showing **{len(fdf):,}** of **{len(df):,}** customers")

# ---------- KPIs ----------
kpis = kpi_summary(fdf)
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Total Customers", f"{kpis['total_customers']:,}")
c2.metric("Predicted Churn Rate", f"{kpis['churn_rate']*100:.1f}%")
c3.metric("Revenue at Risk (yr)", f"${kpis['revenue_at_risk']:,.0f}")
c4.metric("Avg Monthly Revenue", f"${kpis['avg_monthly_revenue']:.2f}")
c5.metric("High-Risk Customers", f"{kpis['high_risk_customers']:,}")
c6.metric("Expected Retention Gain", f"${kpis['expected_retention_savings']:,.0f}",
          help="Assuming 30% successful retention of at-risk customers")

st.divider()

# ---------- Charts ----------
left, right = st.columns(2)
with left:
    st.plotly_chart(churn_distribution(fdf), use_container_width=True)
    st.plotly_chart(churn_by_tenure_bucket(fdf), use_container_width=True)
    st.plotly_chart(monthly_charges_vs_churn(fdf), use_container_width=True)

with right:
    st.plotly_chart(churn_by_category(fdf, "Contract"), use_container_width=True)
    st.plotly_chart(revenue_at_risk_segmentation(fdf), use_container_width=True)
    st.plotly_chart(correlation_heatmap(fdf[["tenure", "MonthlyCharges", "TotalCharges",
                                             "services_count", "charges_per_tenure",
                                             "churn_probability"]]),
                    use_container_width=True)

st.divider()
st.subheader("Top Churn Drivers")
try:
    model, scaler, feature_columns, _ = _artifacts()
    importances = pd.Series(model.feature_importances_, index=feature_columns)
    st.plotly_chart(top_churn_drivers(importances), use_container_width=True)
except Exception as exc:  # noqa: BLE001
    st.warning(f"Could not render feature importances: {exc}")

with st.expander("View raw scored data"):
    st.dataframe(fdf.head(500), use_container_width=True, height=380)
