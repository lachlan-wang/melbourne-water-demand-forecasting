from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline

from model_utils import (
    GAM_FEATURES,
    SKLEARN_FEATURES,
    calculate_metrics,
    load_model_data,
    make_gam_model,
    make_linear_model,
    make_preprocessor,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURE_FILE = PROJECT_ROOT / "data" / "processed" / "model_features.csv"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURE_DATA_DIR = RESULTS_DIR / "figure_data"

VALIDATION_PREDICTIONS_FILE = RESULTS_DIR / "validation_predictions.csv"
VALIDATION_BY_YEAR_FILE = RESULTS_DIR / "validation_by_year.csv"
VALIDATION_METRICS_FILE = RESULTS_DIR / "validation_metrics.csv"
VALIDATION_STABILITY_FILE = RESULTS_DIR / "validation_stability.csv"
TUNING_RESULTS_FILE = RESULTS_DIR / "rf_tuning_results.csv"
BEST_PARAMETERS_FILE = RESULTS_DIR / "rf_best_parameters.json"

RANDOM_STATE = 42
N_ESTIMATORS = 300
VALIDATION_YEARS = list(range(2014, 2019))

def load_development_data(path):
    data = load_model_data(path)
    development = data.loc[
        (data["data_role"] == "development") & data["model_ready"]
    ].copy()
    development = development.sort_values("date").reset_index(drop=True)
    if development["date"].min() != pd.Timestamp("2011-01-01"):
        raise ValueError("Development data does not start on 2011-01-01.")
    if development["date"].max() != pd.Timestamp("2018-12-31"):
        raise ValueError("Development data does not end on 2018-12-31.")
    if development[SKLEARN_FEATURES + GAM_FEATURES + ["supplied_ML"]].isna().any().any():
        raise ValueError("Development model inputs contain missing values.")
    return development


def make_splits(development):
    splits = []
    for validation_year in VALIDATION_YEARS:
        train = development.loc[development["date"].dt.year < validation_year].copy()
        validation = development.loc[
            development["date"].dt.year == validation_year
        ].copy()
        if train.empty or validation.empty:
            raise ValueError(f"Empty training or validation data for {validation_year}.")
        if train["date"].max() >= validation["date"].min():
            raise ValueError(f"Temporal leakage detected in the {validation_year} split.")
        splits.append((validation_year, train, validation))
    return splits


def make_random_forest(parameters):
    regressor = RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        max_depth=parameters["max_depth"],
        min_samples_leaf=parameters["min_samples_leaf"],
        max_features=parameters["max_features"],
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return Pipeline(
        [("preprocess", make_preprocessor(False)), ("model", regressor)]
    )


def metric_record(validation_year, model_name, train, validation, predicted):
    actual = validation["supplied_ML"].to_numpy(dtype=float)
    return {
        "validation_year": validation_year,
        "model": model_name,
        "train_start_date": train["date"].min(),
        "train_end_date": train["date"].max(),
        "validation_start_date": validation["date"].min(),
        "validation_end_date": validation["date"].max(),
        "training_sample_count": len(train),
        "validation_sample_count": len(validation),
        **calculate_metrics(actual, predicted),
    }


def evaluate_baseline_models(splits):
    prediction_tables = []
    yearly_records = []
    baseline_parameters = {
        "max_depth": 12,
        "min_samples_leaf": 5,
        "max_features": 0.8,
    }

    for year, train, validation in splits:
        y_train = train["supplied_ML"].to_numpy(dtype=float)
        actual = validation["supplied_ML"].to_numpy(dtype=float)
        X_train = train[SKLEARN_FEATURES]
        X_validation = validation[SKLEARN_FEATURES]

        seasonal = validation["demand_lag_7"].to_numpy(dtype=float)
        linear = make_linear_model()
        linear.fit(X_train, y_train)
        linear_prediction = linear.predict(X_validation)

        gam = make_gam_model()
        gam.fit(train[GAM_FEATURES].to_numpy(dtype=float), y_train)
        gam_prediction = gam.predict(validation[GAM_FEATURES].to_numpy(dtype=float))

        random_forest = make_random_forest(baseline_parameters)
        random_forest.fit(X_train, y_train)
        forest_prediction = random_forest.predict(X_validation)

        predictions = {
            "SeasonalNaive": seasonal,
            "LinearRegression": linear_prediction,
            "GAM": gam_prediction,
            "RandomForest": forest_prediction,
        }
        prediction_tables.append(
            pd.DataFrame(
                {
                    "date": validation["date"].to_numpy(),
                    "validation_year": year,
                    "actual_supplied_ML": actual,
                    "pred_seasonal_naive": seasonal,
                    "pred_linear_regression": linear_prediction,
                    "pred_gam": gam_prediction,
                    "pred_random_forest": forest_prediction,
                }
            )
        )
        for model_name, predicted in predictions.items():
            yearly_records.append(
                metric_record(year, model_name, train, validation, predicted)
            )

    return pd.concat(prediction_tables, ignore_index=True), yearly_records


def tune_random_forest(splits):
    grid = {
        "max_depth": [8, 12, None],
        "min_samples_leaf": [2, 5, 10],
        "max_features": [0.6, 0.8, 1.0],
    }
    combinations = list(ParameterGrid(grid))
    records = []
    lookup = {}

    for configuration_id, parameters in enumerate(combinations, start=1):
        lookup[configuration_id] = parameters
        pooled_actual = []
        pooled_predicted = []
        yearly_mae = []
        for _, train, validation in splits:
            model = make_random_forest(parameters)
            model.fit(
                train[SKLEARN_FEATURES], train["supplied_ML"].to_numpy(dtype=float)
            )
            predicted = model.predict(validation[SKLEARN_FEATURES])
            actual = validation["supplied_ML"].to_numpy(dtype=float)
            pooled_actual.append(actual)
            pooled_predicted.append(predicted)
            yearly_mae.append(mean_absolute_error(actual, predicted))

        pooled_metrics = calculate_metrics(
            np.concatenate(pooled_actual), np.concatenate(pooled_predicted)
        )
        records.append(
            {
                "configuration_id": configuration_id,
                "n_estimators": N_ESTIMATORS,
                "max_depth": "None"
                if parameters["max_depth"] is None
                else parameters["max_depth"],
                "min_samples_leaf": parameters["min_samples_leaf"],
                "max_features": parameters["max_features"],
                "pooled_MAE_ML": pooled_metrics["MAE_ML"],
                "pooled_RMSE_ML": pooled_metrics["RMSE_ML"],
                "pooled_MAPE_percent": pooled_metrics["MAPE_percent"],
                "pooled_R2": pooled_metrics["R2"],
                "mean_yearly_MAE_ML": np.mean(yearly_mae),
                "std_yearly_MAE_ML": np.std(yearly_mae, ddof=1),
                "worst_yearly_MAE_ML": np.max(yearly_mae),
            }
        )
        print(
            f"RF configuration {configuration_id:02d}/{len(combinations)}: "
            f"pooled MAE {pooled_metrics['MAE_ML']:.3f} ML"
        )

    results = pd.DataFrame(records).sort_values(
        ["pooled_MAE_ML", "pooled_RMSE_ML", "std_yearly_MAE_ML"]
    )
    results = results.reset_index(drop=True)
    results.insert(0, "tuning_rank", np.arange(1, len(results) + 1))
    best_id = int(results.loc[0, "configuration_id"])
    return results, lookup[best_id]


def evaluate_tuned_model(splits, parameters):
    prediction_tables = []
    yearly_records = []
    for year, train, validation in splits:
        model = make_random_forest(parameters)
        model.fit(
            train[SKLEARN_FEATURES], train["supplied_ML"].to_numpy(dtype=float)
        )
        predicted = model.predict(validation[SKLEARN_FEATURES])
        prediction_tables.append(
            pd.DataFrame(
                {
                    "date": validation["date"].to_numpy(),
                    "validation_year": year,
                    "pred_tuned_random_forest": predicted,
                }
            )
        )
        yearly_records.append(
            metric_record(year, "TunedRandomForest", train, validation, predicted)
        )
    return pd.concat(prediction_tables, ignore_index=True), yearly_records


def calculate_pooled_metrics(predictions):
    columns = {
        "SeasonalNaive": "pred_seasonal_naive",
        "LinearRegression": "pred_linear_regression",
        "GAM": "pred_gam",
        "RandomForest": "pred_random_forest",
        "TunedRandomForest": "pred_tuned_random_forest",
    }
    actual = predictions["actual_supplied_ML"].to_numpy(dtype=float)
    records = []
    for model_name, column in columns.items():
        records.append(
            {
                "model": model_name,
                "validation_sample_count": len(predictions),
                "validation_start_date": predictions["date"].min(),
                "validation_end_date": predictions["date"].max(),
                **calculate_metrics(actual, predictions[column]),
            }
        )
    results = pd.DataFrame(records)
    naive_mae = results.loc[results["model"] == "SeasonalNaive", "MAE_ML"].iloc[0]
    results["MAE_improvement_vs_naive_percent"] = (
        (naive_mae - results["MAE_ML"]) / naive_mae * 100.0
    )
    results = results.sort_values("MAE_ML").reset_index(drop=True)
    results.insert(0, "MAE_rank", np.arange(1, len(results) + 1))
    return results


def calculate_stability(yearly_metrics, pooled_metrics):
    winners = yearly_metrics.loc[
        yearly_metrics.groupby("validation_year")["MAE_ML"].idxmin()
    ]
    win_counts = winners["model"].value_counts().rename("years_won")
    stability = (
        yearly_metrics.groupby("model")
        .agg(
            mean_yearly_MAE_ML=("MAE_ML", "mean"),
            std_yearly_MAE_ML=("MAE_ML", "std"),
            best_year_MAE_ML=("MAE_ML", "min"),
            worst_year_MAE_ML=("MAE_ML", "max"),
            mean_yearly_RMSE_ML=("RMSE_ML", "mean"),
            std_yearly_RMSE_ML=("RMSE_ML", "std"),
        )
        .reset_index()
    )
    stability = stability.merge(win_counts, on="model", how="left")
    stability["years_won"] = stability["years_won"].fillna(0).astype(int)
    stability["yearly_MAE_range_ML"] = (
        stability["worst_year_MAE_ML"] - stability["best_year_MAE_ML"]
    )
    stability = stability.merge(
        pooled_metrics[["model", "MAE_rank", "MAE_ML", "RMSE_ML"]].rename(
            columns={"MAE_ML": "pooled_MAE_ML", "RMSE_ML": "pooled_RMSE_ML"}
        ),
        on="model",
        how="left",
        validate="one_to_one",
    )
    return stability.sort_values("MAE_rank").reset_index(drop=True)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    development = load_development_data(FEATURE_FILE)
    splits = make_splits(development)

    predictions, yearly_records = evaluate_baseline_models(splits)
    tuning_results, best_parameters = tune_random_forest(splits)
    tuned_predictions, tuned_records = evaluate_tuned_model(splits, best_parameters)
    predictions = predictions.merge(
        tuned_predictions,
        on=["date", "validation_year"],
        how="left",
        validate="one_to_one",
    ).sort_values("date")
    yearly_metrics = pd.DataFrame(yearly_records + tuned_records).sort_values(
        ["validation_year", "model"]
    )
    pooled_metrics = calculate_pooled_metrics(predictions)
    stability = calculate_stability(yearly_metrics, pooled_metrics)

    tuning_results.to_csv(TUNING_RESULTS_FILE, index=False)
    predictions.to_csv(VALIDATION_PREDICTIONS_FILE, index=False, date_format="%Y-%m-%d")
    yearly_metrics.to_csv(VALIDATION_BY_YEAR_FILE, index=False, date_format="%Y-%m-%d")
    pooled_metrics.to_csv(VALIDATION_METRICS_FILE, index=False, date_format="%Y-%m-%d")
    stability.to_csv(VALIDATION_STABILITY_FILE, index=False)
    figure_model_order = [
        "TunedRandomForest",
        "GAM",
        "LinearRegression",
        "SeasonalNaive",
    ]
    figure_metrics = pooled_metrics.set_index("model").loc[figure_model_order].reset_index()
    figure_metrics["MAE_rank"] = np.arange(1, len(figure_metrics) + 1)
    figure_metrics[["MAE_rank", "model", "MAE_ML", "RMSE_ML"]].to_csv(
        FIGURE_DATA_DIR / "model_mae_comparison.csv", index=False
    )

    best_record = {
        "n_estimators": N_ESTIMATORS,
        "max_depth": best_parameters["max_depth"],
        "min_samples_leaf": best_parameters["min_samples_leaf"],
        "max_features": best_parameters["max_features"],
        "random_state": RANDOM_STATE,
        "validation_years": VALIDATION_YEARS,
        "validation_MAE_ML": float(tuning_results.loc[0, "pooled_MAE_ML"]),
    }
    BEST_PARAMETERS_FILE.write_text(
        json.dumps(best_record, indent=2), encoding="utf-8"
    )

    print("2014-2018 expanding-window validation completed.")
    print(pooled_metrics[["MAE_rank", "model", "MAE_ML", "RMSE_ML"]].to_string(index=False))
    print(f"Best Random Forest parameters: {best_record}")
    print("The 2019 holdout was not used for validation or tuning.")


if __name__ == "__main__":
    main()
