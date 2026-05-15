"""Batch prediction page: upload CSV, score, download."""
from __future__ import annotations
import io
import logging
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app.utils.preprocessing import REQUIRED_COLUMNS, validate_schema  # noqa: E402
from app.utils.prediction import add_predictions, load_artifacts  # noqa: E402
from app.utils.business_metrics import kpi_summary, population_stability_index  # noqa: E402
from app.utils.visualizations import probability_distribution, revenue_at_risk_segmentation  # noqa: E402
from training.config import MODELS_DIR, TRAIN_SAMPLE_PATH  # noqa: E402

logger = logging.getLogger("batch")
st.set_page_config(page_title="Batch Prediction", layout="wide", page_icon="📥")
st.title("Batch Prediction")
st.caption("Upload a CSV with the Telco customer schema to score thousands of customers at once.")

st.markdown("**Required columns:** " + ", ".join(f"`{c}`" for c in REQUIRED_COLUMNS))

up = st.file_uploader("Upload customer CSV", type=["csv"])
if up is None:
    st.info("Awaiting CSV. You can also use `data/raw/Telco-Customer-Churn.csv` shipped with the project.")
    st.stop()

try:
    raw = pd.read_csv(up)
except Exception as exc:  # noqa: BLE001
    st.error(f"Could not read CSV: {exc}")
    st.stop()

st.subheader("Preview")
st.dataframe(raw.head(20), use_container_width=True, height=280)

err = validate_schema(raw)
if err:
    st.error(err)
    st.stop()

with st.spinner("Loading model artifacts and scoring..."):
    try:
        model, scaler, feature_columns, threshold = load_artifacts(MODELS_DIR)
        progress = st.progress(0.0, text="Preparing features")
        progress.progress(0.4, text="Running model")
        scored = add_predictions(raw, model, scaler, feature_columns, threshold)
        progress.progress(1.0, text="Done")
    except Exception as exc:  # noqa: BLE001
        logger.exception("Scoring failed")
        st.error(f"Scoring failed: {exc}")
        st.stop()

st.success(f"Scored {len(scored):,} customers.")

kpis = kpi_summary(scored)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Customers", f"{kpis['total_customers']:,}")
c2.metric("Predicted Churn Rate", f"{kpis['churn_rate']*100:.1f}%")
c3.metric("Revenue at Risk", f"${kpis['revenue_at_risk']:,.0f}")
c4.metric("High-Risk", f"{kpis['high_risk_customers']:,}")

# Drift detection vs training reference
if TRAIN_SAMPLE_PATH.exists():
    try:
        ref = pd.read_parquet(TRAIN_SAMPLE_PATH)
        psi = population_stability_index(ref["MonthlyCharges"].to_numpy(),
                                         scored["MonthlyCharges"].to_numpy())
        if psi > 0.25:
            st.warning(f"⚠️ Drift detected on MonthlyCharges (PSI={psi:.2f}). "
                       "Consider retraining the model on fresher data.")
        else:
            st.caption(f"Distribution stability OK (PSI={psi:.2f}).")
    except Exception:  # noqa: BLE001
        pass

l, r = st.columns(2)
with l:
    st.plotly_chart(probability_distribution(scored["churn_probability"]),
                    use_container_width=True)
with r:
    st.plotly_chart(revenue_at_risk_segmentation(scored), use_container_width=True)

st.subheader("Scored customers")
st.dataframe(scored, use_container_width=True, height=420)

buf = io.StringIO()
scored.to_csv(buf, index=False)
st.download_button("Download predictions as CSV", buf.getvalue(),
                   file_name="churn_predictions.csv", mime="text/csv",
                   type="primary")
