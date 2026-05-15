"""Global and local SHAP explanations."""
from __future__ import annotations
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.utils.preprocessing import clean_dataframe, load_raw  # noqa: E402
from app.utils.feature_engineering import add_engineered_features  # noqa: E402
from app.utils.prediction import load_artifacts, prepare_features  # noqa: E402
from app.utils.shap_utils import (dependence_plot_fig, humanize_feature,
                                  make_explainer, shap_values_for,
                                  summary_plot_fig, top_drivers, waterfall_fig)
from training.config import MODELS_DIR, RAW_CSV  # noqa: E402

logger = logging.getLogger("explain")
st.set_page_config(page_title="Model Explainability", layout="wide", page_icon="🧠")
st.title("Model Explainability (SHAP)")


@st.cache_resource
def _bundle():
    model, scaler, feat, thr = load_artifacts(MODELS_DIR)
    return model, scaler, feat, thr, make_explainer(model)


@st.cache_data(show_spinner=True)
def _sample_X(n: int = 500) -> pd.DataFrame:
    df = clean_dataframe(load_raw(RAW_CSV))
    df = add_engineered_features(df)
    model, scaler, feat, _, _ = _bundle()
    sample = df.sample(min(n, len(df)), random_state=0)
    X = prepare_features(sample, feat, scaler)
    return sample.reset_index(drop=True), X.reset_index(drop=True)


sample_df, X = _sample_X()
model, scaler, feat, thr, explainer = _bundle()

with st.spinner("Computing SHAP values..."):
    expl = shap_values_for(explainer, X)

st.subheader("Global Feature Importance")
mean_abs = pd.Series(np.abs(expl.values).mean(axis=0), index=feat).sort_values(ascending=False).head(15)
st.bar_chart(mean_abs)

st.subheader("SHAP Summary Plot")
st.pyplot(summary_plot_fig(expl), use_container_width=True)

st.subheader("Dependence Plot")
top_feature = mean_abs.index[0]
chosen = st.selectbox("Feature", mean_abs.index.tolist(), index=0)
try:
    st.pyplot(dependence_plot_fig(expl, chosen), use_container_width=True)
except Exception as exc:  # noqa: BLE001
    st.warning(f"Dependence plot unavailable for {chosen}: {exc}")

st.divider()
st.subheader("Individual Prediction Explanation")
idx = st.slider("Customer index", 0, len(sample_df) - 1, 0)
row = sample_df.iloc[idx]
st.write(row.to_frame().T)

drivers = top_drivers(expl.values[idx], feat, top_n=3)
st.markdown("**Top 3 churn drivers for this customer:**")
for name, val, direction in drivers:
    color = "🔴" if direction == "increases" else "🟢"
    st.markdown(f"- {color} **{name}** {direction} churn risk by `{abs(val):.3f}`")

st.pyplot(waterfall_fig(expl, idx), use_container_width=True)

st.info("Month-to-month contracts, fiber-optic internet, and electronic-check payments "
        "are among the strongest churn drivers in this dataset.")
