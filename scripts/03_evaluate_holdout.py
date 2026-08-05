from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline

from model_utils import (
    GAM_FEATURES,
    SKLEARN_FEATURES,
    calculate_metrics,
    load_model_data as load_prepared_model_data,
    make_gam_model,
    make_linear_model,
    make_preprocessor,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURE_FILE = PROJECT_ROOT / "data" / "processed" / "model_features.csv"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURE_DATA_DIR = RESULTS_DIR / "figure_data"
BEST_PARAMETERS_FILE = RESULTS_DIR / "rf_best_parameters.json"
PREDICTIONS_FILE = RESULTS_DIR / "holdout_predictions_2019.csv"
METRICS_FILE = RESULTS_DIR / "holdout_metrics.csv"
MONTHLY_METRICS_FILE = RESULTS_DIR / "holdout_monthly_mae.csv"

def load_model_data(path):
    data = load_prepared_model_data(path)
    development = data.loc[
        (data["data_role"] == "development") & data["model_ready"]
    ].copy()
    holdout = data.loc[(data["data_role"] == "test") & data["model_ready"]].copy()
    development = development.sort_values("date").reset_index(drop=True)
    holdout = holdout.sort_values("date").reset_index(drop=True)

    if development["date"].min() != pd.Timestamp("2011-01-01"):
        raise ValueError("Development data does not start on 2011-01-01.")
    if development["date"].max() != pd.Timestamp("2018-12-31"):
        raise ValueError("Development data does not end on 2018-12-31.")
    if holdout["date"].min() != pd.Timestamp("2019-01-01"):
        raise ValueError("Holdout data does not start on 2019-01-01.")
    if holdout["date"].max() != pd.Timestamp("2019-12-31") or len(holdout) != 365:
        raise ValueError("The 2019 holdout must contain all 365 days.")
    if development["date"].max() >= holdout["date"].min():
        raise ValueError("Development and holdout periods overlap.")
    if development[SKLEARN_FEATURES + GAM_FEATURES].isna().any().any():
        raise ValueError("Development model inputs contain missing values.")
    if holdout[SKLEARN_FEATURES + GAM_FEATURES].isna().any().any():
        raise ValueError("Holdout model inputs contain missing values.")
    return development, holdout


def make_tuned_random_forest(parameters):
    regressor = RandomForestRegressor(
        n_estimators=parameters["n_estimators"],
        max_depth=parameters["max_depth"],
        min_samples_leaf=parameters["min_samples_leaf"],
        max_features=parameters["max_features"],
        random_state=parameters["random_state"],
        n_jobs=-1,
    )
    return Pipeline(
        [("preprocess", make_preprocessor(False)), ("model", regressor)]
    )


def calculate_holdout_metrics(actual, predicted):
    metrics = calculate_metrics(actual, predicted)
    negative_count = metrics.pop("negative_prediction_count")
    errors = np.asarray(predicted, dtype=float) - np.asarray(actual, dtype=float)
    metrics["MeanError_ML"] = np.mean(errors)
    metrics["negative_prediction_count"] = negative_count
    return metrics


def load_best_parameters(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Best parameter file not found: {path}. Run 02_validate_and_tune.py first."
        )
    parameters = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "n_estimators": 300,
        "max_depth": None,
        "min_samples_leaf": 2,
        "max_features": 0.8,
        "random_state": 42,
    }
    actual = {key: parameters.get(key) for key in expected}
    if actual != expected:
        raise ValueError(
            f"Tuned parameters differ from the preselected specification: {actual}"
        )
    return actual


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    development, holdout = load_model_data(FEATURE_FILE)
    parameters = load_best_parameters(BEST_PARAMETERS_FILE)

    y_train = development["supplied_ML"].to_numpy(dtype=float)
    y_holdout = holdout["supplied_ML"].to_numpy(dtype=float)
    X_train = development[SKLEARN_FEATURES]
    X_holdout = holdout[SKLEARN_FEATURES]

    seasonal_prediction = holdout["demand_lag_7"].to_numpy(dtype=float)
    linear_model = make_linear_model()
    linear_model.fit(X_train, y_train)
    linear_prediction = linear_model.predict(X_holdout)

    gam_model = make_gam_model()
    gam_model.fit(development[GAM_FEATURES].to_numpy(dtype=float), y_train)
    gam_prediction = gam_model.predict(holdout[GAM_FEATURES].to_numpy(dtype=float))

    tuned_model = make_tuned_random_forest(parameters)
    tuned_model.fit(X_train, y_train)
    tuned_prediction = tuned_model.predict(X_holdout)

    predictions = {
        "SeasonalNaive": seasonal_prediction,
        "LinearRegression": linear_prediction,
        "GAM": gam_prediction,
        "TunedRandomForest": tuned_prediction,
    }
    prediction_table = pd.DataFrame(
        {
            "date": holdout["date"].to_numpy(),
            "actual_supplied_ML": y_holdout,
            "pred_seasonal_naive": seasonal_prediction,
            "pred_linear_regression": linear_prediction,
            "pred_gam": gam_prediction,
            "pred_tuned_random_forest": tuned_prediction,
        }
    )
    prediction_table["error_seasonal_naive_ML"] = seasonal_prediction - y_holdout
    prediction_table["error_linear_regression_ML"] = linear_prediction - y_holdout
    prediction_table["error_gam_ML"] = gam_prediction - y_holdout
    prediction_table["error_tuned_random_forest_ML"] = tuned_prediction - y_holdout
    prediction_table["month"] = prediction_table["date"].dt.month

    metric_records = []
    for model_name, predicted in predictions.items():
        metric_records.append(
            {
                "model": model_name,
                "preselected_primary_model": model_name == "TunedRandomForest",
                "training_sample_count": len(development),
                "test_sample_count": len(holdout),
                "train_start_date": development["date"].min(),
                "train_end_date": development["date"].max(),
                "test_start_date": holdout["date"].min(),
                "test_end_date": holdout["date"].max(),
                **calculate_holdout_metrics(y_holdout, predicted),
            }
        )
    metrics = pd.DataFrame(metric_records)
    naive_mae = metrics.loc[metrics["model"] == "SeasonalNaive", "MAE_ML"].iloc[0]
    metrics["MAE_improvement_vs_naive_percent"] = (
        (naive_mae - metrics["MAE_ML"]) / naive_mae * 100.0
    )
    metrics = metrics.sort_values("MAE_ML").reset_index(drop=True)
    metrics.insert(0, "descriptive_test_MAE_rank", np.arange(1, len(metrics) + 1))

    prediction_columns = {
        "SeasonalNaive": "pred_seasonal_naive",
        "LinearRegression": "pred_linear_regression",
        "GAM": "pred_gam",
        "TunedRandomForest": "pred_tuned_random_forest",
    }
    monthly_records = []
    for month in range(1, 13):
        month_data = prediction_table.loc[prediction_table["month"] == month]
        if month_data.empty:
            raise ValueError(f"No holdout data found for month {month}.")
        for model_name, column in prediction_columns.items():
            monthly_records.append(
                {
                    "month": month,
                    "model": model_name,
                    "sample_count": len(month_data),
                    "MAE_ML": mean_absolute_error(
                        month_data["actual_supplied_ML"], month_data[column]
                    ),
                }
            )
    monthly_metrics = pd.DataFrame(monthly_records).sort_values(["month", "model"])

    prediction_table.to_csv(PREDICTIONS_FILE, index=False, date_format="%Y-%m-%d")
    metrics.to_csv(METRICS_FILE, index=False, date_format="%Y-%m-%d")
    monthly_metrics.to_csv(MONTHLY_METRICS_FILE, index=False)

    figure_table = prediction_table[
        ["date", "actual_supplied_ML", "pred_tuned_random_forest"]
    ].rename(
        columns={
            "actual_supplied_ML": "observed_ML",
            "pred_tuned_random_forest": "predicted_ML",
        }
    )
    figure_table.to_csv(
        FIGURE_DATA_DIR / "holdout_timeseries_2019.csv",
        index=False,
        date_format="%Y-%m-%d",
    )
    figure_table[["observed_ML", "predicted_ML"]].to_csv(
        FIGURE_DATA_DIR / "holdout_observed_vs_predicted.csv", index=False
    )

    primary = metrics.loc[metrics["preselected_primary_model"]].iloc[0]
    print("2019 independent holdout evaluation completed.")
    print(f"Tuned Random Forest MAE: {primary['MAE_ML']:.2f} ML")
    print(f"Tuned Random Forest RMSE: {primary['RMSE_ML']:.2f} ML")
    print(f"Tuned Random Forest MAPE: {primary['MAPE_percent']:.2f}%")
    print(f"Tuned Random Forest R2: {primary['R2']:.4f}")
    print(f"Negative predictions: {int(primary['negative_prediction_count'])}")
    print("Holdout results were not used for model selection or tuning.")


if __name__ == "__main__":
    main()
