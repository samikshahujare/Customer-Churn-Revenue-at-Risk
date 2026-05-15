"""Plotly visualization helpers for the dashboard."""
from __future__ import annotations
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

PLOTLY_TEMPLATE = "plotly_dark"
ACCENT = "#7C5CFF"
ACCENT2 = "#22D3EE"
DANGER = "#F43F5E"


def churn_distribution(df: pd.DataFrame) -> go.Figure:
    counts = df["Churn"].value_counts().reset_index()
    counts.columns = ["Churn", "count"]
    fig = px.pie(counts, names="Churn", values="count", hole=0.55,
                 color_discrete_sequence=[ACCENT, DANGER],
                 template=PLOTLY_TEMPLATE)
    fig.update_layout(title="Churn Distribution", height=360)
    return fig


def churn_by_category(df: pd.DataFrame, column: str) -> go.Figure:
    grp = df.groupby([column, "Churn"]).size().reset_index(name="count")
    fig = px.bar(grp, x=column, y="count", color="Churn", barmode="group",
                 color_discrete_map={"Yes": DANGER, "No": ACCENT},
                 template=PLOTLY_TEMPLATE)
    fig.update_layout(title=f"Churn by {column}", height=360)
    return fig


def churn_by_tenure_bucket(df: pd.DataFrame) -> go.Figure:
    if "tenure_bucket" not in df.columns:
        return go.Figure()
    order = ["0-6m", "6-12m", "1-2y", "2-4y", "4-6y", "6y+"]
    grp = df.groupby("tenure_bucket")["Churn"].apply(lambda s: (s == "Yes").mean()).reindex(order).reset_index()
    grp.columns = ["tenure_bucket", "churn_rate"]
    fig = px.bar(grp, x="tenure_bucket", y="churn_rate",
                 color="churn_rate", color_continuous_scale="magma",
                 template=PLOTLY_TEMPLATE)
    fig.update_layout(title="Churn Rate by Tenure Bucket", yaxis_tickformat=".0%", height=360)
    return fig


def monthly_charges_vs_churn(df: pd.DataFrame) -> go.Figure:
    fig = px.box(df, x="Churn", y="MonthlyCharges", color="Churn",
                 color_discrete_map={"Yes": DANGER, "No": ACCENT},
                 template=PLOTLY_TEMPLATE)
    fig.update_layout(title="Monthly Charges vs Churn", height=360)
    return fig


def correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    num = df.select_dtypes(include="number")
    if num.shape[1] < 2:
        return go.Figure()
    corr = num.corr()
    fig = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns,
                                    colorscale="Viridis"))
    fig.update_layout(title="Correlation Heatmap", template=PLOTLY_TEMPLATE, height=420)
    return fig


def revenue_at_risk_segmentation(df: pd.DataFrame) -> go.Figure:
    if "risk_category" not in df.columns or "revenue_at_risk_annual" not in df.columns:
        return go.Figure()
    grp = df.groupby("risk_category")["revenue_at_risk_annual"].sum().reset_index()
    order = ["Low", "Medium", "High", "Critical"]
    grp["risk_category"] = pd.Categorical(grp["risk_category"], categories=order, ordered=True)
    grp = grp.sort_values("risk_category")
    fig = px.bar(grp, x="risk_category", y="revenue_at_risk_annual",
                 color="risk_category",
                 color_discrete_map={"Low": "#22C55E", "Medium": "#EAB308",
                                     "High": "#F97316", "Critical": "#EF4444"},
                 template=PLOTLY_TEMPLATE)
    fig.update_layout(title="Annual Revenue at Risk by Segment", height=360,
                      yaxis_title="Revenue at risk ($)")
    return fig


def top_churn_drivers(importances: pd.Series, top_n: int = 12) -> go.Figure:
    s = importances.sort_values(ascending=True).tail(top_n)
    fig = px.bar(x=s.values, y=s.index, orientation="h",
                 color=s.values, color_continuous_scale="plasma",
                 template=PLOTLY_TEMPLATE)
    fig.update_layout(title="Top Churn Drivers (Model Importance)",
                      xaxis_title="Importance", yaxis_title="Feature", height=460)
    return fig


def gauge_chart(value: float, title: str = "Churn Probability") -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value * 100,
        number={"suffix": "%"},
        title={"text": title},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": ACCENT},
            "steps": [
                {"range": [0, 25], "color": "#16341F"},
                {"range": [25, 50], "color": "#3B3413"},
                {"range": [50, 75], "color": "#3B1F12"},
                {"range": [75, 100], "color": "#3B121C"},
            ],
        },
    ))
    fig.update_layout(template=PLOTLY_TEMPLATE, height=320)
    return fig


def probability_distribution(probs) -> go.Figure:
    fig = px.histogram(x=probs, nbins=40, template=PLOTLY_TEMPLATE,
                       color_discrete_sequence=[ACCENT2])
    fig.update_layout(title="Churn Probability Distribution",
                      xaxis_title="Probability", yaxis_title="Customers", height=320)
    return fig
