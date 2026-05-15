"""Interactive what-if simulator for a single customer."""
from __future__ import annotations
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app.utils.preprocessing import clean_dataframe, load_raw  # noqa: E402
from app.utils.feature_engineering import add_engineered_features  # noqa: E402
from app.utils.prediction import (categorize_risk, load_artifacts,
                                  prepare_features, predict_proba)
from app.utils.business_metrics import retention_recommendations  # noqa: E402
from app.utils.shap_utils import (humanize_feature, make_explainer,
                                  shap_values_for, top_drivers, waterfall_fig)
from app.utils.visualizations import gauge_chart  # noqa: E402
from training.config import MODELS_DIR, RAW_CSV  # noqa: E402

logger = logging.getLogger("whatif")
st.set_page_config(page_title="What-If Analysis", layout="wide", page_icon="🧪")
st.title("What-If Analysis")
st.caption("Adjust customer attributes and see how churn risk and revenue impact change.")


@st.cache_data
def _customers() -> pd.DataFrame:
    return clean_dataframe(load_raw(RAW_CSV))


@st.cache_resource
def _bundle():
    model, scaler, feat, thr = load_artifacts(MODELS_DIR)
    explainer = make_explainer(model)
    return model, scaler, feat, thr, explainer


df = _customers()
model, scaler, feat, thr, explainer = _bundle()

ids = df["customerID"].tolist()
default_id = ids[0]
selected = st.selectbox("Pick a customer", ids, index=ids.index(default_id))
base = df[df["customerID"] == selected].iloc[0].copy()

st.markdown("### Adjust attributes")
col1, col2, col3 = st.columns(3)
with col1:
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"],
                            index=["Month-to-month", "One year", "Two year"].index(base["Contract"]))
    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"],
                            index=["DSL", "Fiber optic", "No"].index(base["InternetService"]))
    tech = st.selectbox("Tech Support", ["No", "Yes", "No internet service"],
                        index=["No", "Yes", "No internet service"].index(base["TechSupport"]))
with col2:
    monthly = st.slider("Monthly Charges ($)", 10.0, 200.0, float(base["MonthlyCharges"]), 0.5)
    tenure = st.slider("Tenure (months)", 0, 72, int(base["tenure"]))
with col3:
    payment = st.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        index=["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"].index(base["PaymentMethod"]),
    )

modified = base.copy()
modified["Contract"] = contract
modified["InternetService"] = internet
modified["TechSupport"] = tech
modified["MonthlyCharges"] = monthly
modified["tenure"] = tenure
modified["PaymentMethod"] = payment
modified["TotalCharges"] = monthly * max(tenure, 1)

frame_base = add_engineered_features(pd.DataFrame([base]))
frame_mod = add_engineered_features(pd.DataFrame([modified]))

X_base = prepare_features(frame_base, feat, scaler)
X_mod = prepare_features(frame_mod, feat, scaler)

p_base = float(predict_proba(model, X_base)[0])
p_mod = float(predict_proba(model, X_mod)[0])

g1, g2 = st.columns(2)
with g1:
    st.plotly_chart(gauge_chart(p_base, "Original Churn Probability"), use_container_width=True)
with g2:
    st.plotly_chart(gauge_chart(p_mod, "Simulated Churn Probability"), use_container_width=True)

delta_prob = p_base - p_mod
revenue_saved = max(delta_prob, 0) * monthly * 12
m1, m2, m3, m4 = st.columns(4)
m1.metric("Original Risk", categorize_risk(p_base))
m2.metric("Simulated Risk", categorize_risk(p_mod), f"{(p_mod - p_base)*100:+.1f} pp")
m3.metric("Annual Revenue Impact", f"${(p_mod - p_base) * monthly * 12:+,.0f}")
m4.metric("Estimated Retention Saving", f"${revenue_saved:,.0f}")

st.divider()
st.subheader("Live SHAP Explanation (Simulated)")
try:
    expl = shap_values_for(explainer, X_mod)
    drivers = top_drivers(expl.values[0], feat, top_n=3)
    bullets = [f"- **{name}** {direction} churn risk by {abs(val):.3f}" for name, val, direction in drivers]
    st.markdown("\n".join(bullets))
    st.pyplot(waterfall_fig(expl, 0), use_container_width=True)
except Exception as exc:  # noqa: BLE001
    st.warning(f"Could not render SHAP for this row: {exc}")

st.divider()
st.subheader("Recommended Retention Plays")
for r in retention_recommendations(modified):
    st.markdown(f"- {r}")
