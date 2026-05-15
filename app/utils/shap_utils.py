"""SHAP explainability helpers."""
from __future__ import annotations
import logging
from functools import lru_cache
from typing import Tuple
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def make_explainer(model):
    """Build a TreeExplainer for an XGBoost model."""
    return shap.TreeExplainer(model)


def shap_values_for(explainer, X: pd.DataFrame):
    """Compute SHAP values; returns the shap.Explanation object."""
    return explainer(X)


def humanize_feature(name: str) -> str:
    """Make engineered/dummy feature names readable for business users."""
    mapping = {
        "Contract_Month-to-month": "Month-to-month contract",
        "Contract_Two year": "Two-year contract",
        "Contract_One year": "One-year contract",
        "InternetService_Fiber optic": "Fiber-optic internet",
        "InternetService_DSL": "DSL internet",
        "PaymentMethod_Electronic check": "Electronic-check payment",
        "tenure": "Tenure (months)",
        "MonthlyCharges": "Monthly charges",
        "TotalCharges": "Total charges",
        "services_count": "Number of services",
        "charges_per_tenure": "Avg charges per month of tenure",
        "auto_payment_flag": "Auto-pay enabled",
    }
    return mapping.get(name, name.replace("_", " "))


def top_drivers(shap_row: np.ndarray, feature_names, top_n: int = 3):
    """Return list of (feature, value, direction) for the top N abs-impact features."""
    pairs = sorted(zip(feature_names, shap_row), key=lambda kv: abs(kv[1]), reverse=True)[:top_n]
    return [(humanize_feature(f), float(v), "increases" if v > 0 else "reduces") for f, v in pairs]


def summary_plot_fig(shap_explanation, max_display: int = 15):
    """Return a matplotlib figure for the SHAP summary plot."""
    plt.close("all")
    shap.summary_plot(shap_explanation, show=False, max_display=max_display, plot_size=(8, 5))
    fig = plt.gcf()
    return fig


def dependence_plot_fig(shap_explanation, feature: str):
    plt.close("all")
    shap.plots.scatter(shap_explanation[:, feature], show=False)
    return plt.gcf()


def waterfall_fig(shap_explanation, index: int = 0):
    plt.close("all")
    shap.plots.waterfall(shap_explanation[index], show=False, max_display=12)
    return plt.gcf()
