"""Home page for the Customer Churn & Revenue-at-Risk Analytics Platform."""
from __future__ import annotations
import json
import logging
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.config import ASSETS_DIR, METRICS_PATH  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("home")

st.set_page_config(
    page_title="Churn & Revenue-at-Risk Platform",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS = """
<style>
.block-container { padding-top: 2rem; }
.kpi {
  background: linear-gradient(135deg, #141A2E 0%, #1B2240 100%);
  border: 1px solid #232A45;
  border-radius: 14px; padding: 18px 20px;
  box-shadow: 0 6px 24px rgba(0,0,0,0.25);
}
.kpi h4 { color:#9AA3C0; margin:0; font-weight:500; font-size:0.85rem; letter-spacing:.04em; text-transform:uppercase;}
.kpi .v { color:#E6E9F2; font-size:1.7rem; font-weight:700; margin-top:6px;}
.kpi .d { color:#7C5CFF; font-size:.85rem; margin-top:4px;}
.hero {
  background: radial-gradient(1200px 400px at 10% 0%, rgba(124,92,255,0.25), transparent 60%),
              linear-gradient(135deg, #0F1530 0%, #141A2E 100%);
  border-radius: 18px; padding: 28px 32px; border:1px solid #232A45;
}
.badge { display:inline-block; padding:3px 10px; border-radius:999px; font-size:.75rem; font-weight:600;}
.badge-low{background:#0E2A18;color:#22C55E;border:1px solid #1F4A2F;}
.badge-med{background:#2A220E;color:#EAB308;border:1px solid #4A3F1F;}
.badge-high{background:#2A170E;color:#F97316;border:1px solid #4A2D1F;}
.badge-crit{background:#2A0E16;color:#EF4444;border:1px solid #4A1F2D;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

logo = ASSETS_DIR / "logo.png"
with st.sidebar:
    if logo.exists():
        st.image(str(logo), use_container_width=True)
    st.markdown("### Navigation")
    st.write("Use the pages in the sidebar:")
    st.markdown("- **Dashboard** — executive KPIs & charts")
    st.markdown("- **Batch Prediction** — upload a CSV")
    st.markdown("- **What-If Analysis** — simulate retention plays")
    st.markdown("- **Model Explainability** — SHAP")

st.markdown(
    """
    <div class="hero">
      <h1 style="margin:0">Customer Churn & Revenue-at-Risk Platform</h1>
      <p style="color:#9AA3C0; margin-top:8px; font-size:1.05rem;">
        Predict churn, quantify revenue at risk, and prescribe retention plays
        powered by XGBoost and SHAP.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")

cols = st.columns(4)
metrics = {}
if METRICS_PATH.exists():
    metrics = json.loads(METRICS_PATH.read_text())

def kpi(col, label, value, delta=""):
    col.markdown(f"<div class='kpi'><h4>{label}</h4><div class='v'>{value}</div><div class='d'>{delta}</div></div>", unsafe_allow_html=True)

kpi(cols[0], "Model ROC-AUC", f"{metrics.get('roc_auc', 0):.3f}", "XGBoost (5-fold CV)")
kpi(cols[1], "Model F1", f"{metrics.get('f1', 0):.3f}", f"Threshold {metrics.get('optimal_threshold_youden_j', 0.5):.2f}")
kpi(cols[2], "Precision", f"{metrics.get('precision', 0):.3f}", "On hold-out test set")
kpi(cols[3], "Recall", f"{metrics.get('recall', 0):.3f}", "On hold-out test set")

st.write("")
st.subheader("How to use this platform")
st.markdown(
    """
1. Open **Dashboard** for executive KPIs and revenue-at-risk by segment.
2. Use **Batch Prediction** to score a CSV of customers and download results.
3. Run **What-If Analysis** to simulate retention plays for a single customer.
4. Inspect **Model Explainability** to see global and local SHAP explanations.
"""
)

arch = ASSETS_DIR / "architecture_diagram.png"
if arch.exists():
    with st.expander("System Architecture", expanded=False):
        st.image(str(arch), use_container_width=True)
