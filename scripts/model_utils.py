from pathlib import Path

import numpy as np
import pandas as pd
from pygam import LinearGAM, f, s
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC_FEATURES = [
    "temp_mean_c",
    "temp_range_c",
    "precipitation_log1p",
    "demand_lag_1",
    "demand_lag_7",
    "demand_rolling_mean_7",
    "precipitation_previous_7d_sum",
    "annual_sin",
    "annual_cos",
]
CATEGORICAL_FEATURES = ["day_of_week", "is_public_holiday"]
SKLEARN_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
GAM_FEATURES = [
    "temp_mean_c",
    "temp_range_c",
    "precipitation_log1p",
    "demand_lag_1",
    "demand_lag_7",
    "demand_rolling_mean_7",
    "precipitation_previous_7d_sum",
    "annual_phase",
    "day_of_week",
    "is_public_holiday",
]


def load_model_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Model feature file not found: {path}")

    data = pd.read_csv(path)
    required = {
        "date",
        "supplied_ML",
        "data_role",
        "model_ready",
        "day_of_year",
        *SKLEARN_FEATURES,
    }
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Model feature file is missing columns: {sorted(missing)}")

    data["date"] = pd.to_datetime(data["date"], errors="raise")
    data["data_role"] = data["data_role"].astype(str).str.strip().str.lower()
    if data["model_ready"].dtype != bool:
        ready = data["model_ready"].astype(str).str.strip().str.lower()
        data["model_ready"] = ready.map({"true": True, "false": False})
    if data["model_ready"].isna().any():
        raise ValueError("model_ready contains unrecognised values.")

    numeric_columns = list(set(SKLEARN_FEATURES + ["day_of_year", "supplied_ML"]))
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="raise")
    days_in_year = np.where(data["date"].dt.is_leap_year, 366.0, 365.0)
    data["annual_phase"] = (data["day_of_year"] - 1.0) / days_in_year
    return data


def make_preprocessor(scale_numeric: bool) -> ColumnTransformer:
    numeric_transformer = StandardScaler() if scale_numeric else "passthrough"
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, NUMERIC_FEATURES),
            (
                "categorical",
                OneHotEncoder(
                    drop="first", handle_unknown="ignore", sparse_output=False
                ),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )


def make_linear_model() -> Pipeline:
    return Pipeline(
        [("preprocess", make_preprocessor(True)), ("model", LinearRegression())]
    )


def make_gam_model() -> LinearGAM:
    terms = (
        s(0, n_splines=12)
        + s(1, n_splines=8)
        + s(2, n_splines=8)
        + s(3, n_splines=12)
        + s(4, n_splines=12)
        + s(5, n_splines=12)
        + s(6, n_splines=8)
        + s(7, n_splines=20, basis="cp", edge_knots=[0.0, 1.0])
        + f(8)
        + f(9)
    )
    return LinearGAM(terms=terms, max_iter=200, tol=0.0001)


def calculate_metrics(actual, predicted) -> dict:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    return {
        "MAE_ML": mean_absolute_error(actual, predicted),
        "RMSE_ML": np.sqrt(mean_squared_error(actual, predicted)),
        "MAPE_percent": mean_absolute_percentage_error(actual, predicted) * 100.0,
        "R2": r2_score(actual, predicted),
        "negative_prediction_count": int(np.sum(predicted < 0)),
    }
