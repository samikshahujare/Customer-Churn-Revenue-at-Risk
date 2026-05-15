"""Central configuration for the churn ML pipeline."""
from __future__ import annotations
from pathlib import Path

ROOT_DIR: Path = Path(__file__).resolve().parents[1]
DATA_RAW_DIR: Path = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR: Path = ROOT_DIR / "data" / "processed"
MODELS_DIR: Path = ROOT_DIR / "models"
ASSETS_DIR: Path = ROOT_DIR / "assets"

RAW_CSV: Path = DATA_RAW_DIR / "Telco-Customer-Churn.csv"

MODEL_PATH: Path = MODELS_DIR / "churn_model.pkl"
SCALER_PATH: Path = MODELS_DIR / "scaler.pkl"
FEATURES_PATH: Path = MODELS_DIR / "feature_columns.pkl"
METRICS_PATH: Path = MODELS_DIR / "metrics.json"
THRESHOLD_PATH: Path = MODELS_DIR / "threshold.json"
TRAIN_SAMPLE_PATH: Path = MODELS_DIR / "train_reference.parquet"

TARGET_COLUMN: str = "Churn"
ID_COLUMN: str = "customerID"
RANDOM_STATE: int = 42
TEST_SIZE: float = 0.2

NUMERIC_COLUMNS = ["tenure", "MonthlyCharges", "TotalCharges"]
ENGINEERED_NUMERIC = [
    "services_count",
    "charges_per_tenure",
    "auto_payment_flag",
]
CATEGORICAL_COLUMNS = [
    "gender", "SeniorCitizen", "Partner", "Dependents",
    "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
    "tenure_bucket", "avg_monthly_spend_category",
]

XGB_PARAM_GRID = {
    "max_depth": [3, 5, 7],
    "learning_rate": [0.05, 0.1],
    "n_estimators": [200, 400],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
}

# Reduced grid for faster CI / sandbox training
XGB_PARAM_GRID_FAST = {
    "max_depth": [4, 6],
    "learning_rate": [0.1],
    "n_estimators": [300],
    "subsample": [0.9],
    "colsample_bytree": [0.9],
}
