from pathlib import Path

import holidays
import numpy as np
import pandas as pd
import xarray as xr
from netCDF4 import Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
ERA5_DIR = RAW_DATA_DIR / "era5"
SUPPLY_FILE = RAW_DATA_DIR / "melbourne_water_daily_supply.csv"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
WEATHER_OUTPUT = PROCESSED_DATA_DIR / "era5_daily.csv"
FEATURE_OUTPUT = PROCESSED_DATA_DIR / "model_features.csv"

DATA_START = pd.Timestamp("2010-11-01")
DATA_END = pd.Timestamp("2019-12-31")
DEVELOPMENT_START = pd.Timestamp("2011-01-01")
TEST_START = pd.Timestamp("2019-01-01")


def require_columns(data, required, label):
    missing = set(required) - set(data.columns)
    if missing:
        raise ValueError(f"{label} is missing required columns: {sorted(missing)}")


def load_supply_data(path):
    if not path.exists():
        raise FileNotFoundError(f"Supply data not found: {path}")

    data = pd.read_csv(path)
    require_columns(data, ["recorddate", "supplied_ML"], "Supply data")
    data["date"] = pd.to_datetime(
        data["recorddate"], format="%m/%d/%Y %I:%M:%S %p", errors="raise"
    ).dt.normalize()
    data["supplied_ML"] = pd.to_numeric(data["supplied_ML"], errors="raise")
    data = data.loc[data["date"].between(DATA_START, DATA_END)].copy()
    data = data.sort_values("date").reset_index(drop=True)

    if data["date"].duplicated().any():
        raise ValueError("Supply data contains duplicate dates.")
    expected_dates = pd.date_range(DATA_START, DATA_END, freq="D")
    missing_dates = expected_dates.difference(pd.DatetimeIndex(data["date"]))
    if len(missing_dates):
        raise ValueError(f"Supply data is missing {len(missing_dates)} dates.")
    if data["supplied_ML"].isna().any() or (data["supplied_ML"] <= 0).any():
        raise ValueError("Supply volume contains missing or non-positive values.")

    return data[["date", "supplied_ML"]].copy()


def add_calendar_features(data):
    data = data.copy()
    victoria_holidays = holidays.country_holidays(
        country="AU", subdiv="VIC", years=range(2010, 2020), observed=True
    )
    holiday_names = data["date"].dt.date.map(victoria_holidays.get)

    data["is_public_holiday"] = holiday_names.notna().astype("int8")
    data["public_holiday_name"] = holiday_names.fillna("")
    data["day_of_week"] = data["date"].dt.dayofweek
    data["day_name"] = data["date"].dt.day_name()
    data["is_weekend"] = data["day_of_week"].isin([5, 6]).astype("int8")
    data["month"] = data["date"].dt.month
    data["day_of_year"] = data["date"].dt.dayofyear
    data["year"] = data["date"].dt.year

    data["data_role"] = "development"
    data.loc[data["date"] < DEVELOPMENT_START, "data_role"] = "warmup"
    data.loc[data["date"] >= TEST_START, "data_role"] = "test"

    return data


def read_netcdf(path):
    # An in-memory Dataset avoids Windows path-encoding problems in NetCDF-C.
    netcdf = Dataset("inmemory.nc", mode="r", memory=path.read_bytes())
    store = xr.backends.NetCDF4DataStore(netcdf)
    with xr.open_dataset(store) as dataset:
        return dataset.load()


def extract_hourly_series(dataset, variable_name, output_name):
    time_name = "valid_time" if "valid_time" in dataset.coords else "time"
    if time_name not in dataset.coords:
        raise ValueError(f"No time coordinate found for {variable_name}.")

    values = dataset[variable_name]
    spatial_dims = [dim for dim in ("latitude", "longitude") if dim in values.dims]
    if len(spatial_dims) != 2:
        raise ValueError(f"{variable_name} does not contain latitude and longitude dimensions.")
    values = values.mean(dim=spatial_dims, skipna=True)
    table = values.to_dataframe(name=output_name).reset_index()
    return table[[time_name, output_name]].rename(columns={time_name: "valid_time"})


def load_hourly_weather(root):
    if not root.exists():
        raise FileNotFoundError(f"ERA5 directory not found: {root}")

    yearly_tables = []
    for year in range(2010, 2020):
        year_dir = root / str(year)
        files = sorted(year_dir.glob("*.nc"))
        if not files:
            raise FileNotFoundError(f"No NetCDF files found for {year}: {year_dir}")

        temperature = None
        precipitation = None
        for path in files:
            dataset = read_netcdf(path)
            if "t2m" in dataset.data_vars:
                if temperature is not None:
                    raise ValueError(f"Multiple t2m files found for {year}.")
                temperature = extract_hourly_series(dataset, "t2m", "temperature_k")
            if "tp" in dataset.data_vars:
                if precipitation is not None:
                    raise ValueError(f"Multiple tp files found for {year}.")
                precipitation = extract_hourly_series(dataset, "tp", "precipitation_m")

        if temperature is None or precipitation is None:
            raise ValueError(f"Both t2m and tp are required for {year}.")

        table = pd.merge(
            temperature, precipitation, on="valid_time", how="inner", validate="one_to_one"
        )
        table["valid_time"] = pd.to_datetime(table["valid_time"], errors="raise")
        table = table.sort_values("valid_time").reset_index(drop=True)
        expected_hours = 8784 if pd.Timestamp(f"{year}-01-01").is_leap_year else 8760
        if len(table) != expected_hours:
            raise ValueError(
                f"Unexpected ERA5 hour count for {year}: expected {expected_hours}, found {len(table)}."
            )
        yearly_tables.append(table)

    hourly = pd.concat(yearly_tables, ignore_index=True).sort_values("valid_time")
    hourly = hourly.reset_index(drop=True)
    if hourly["valid_time"].duplicated().any():
        raise ValueError("ERA5 data contains duplicate UTC timestamps.")
    expected_index = pd.date_range(
        hourly["valid_time"].min(), hourly["valid_time"].max(), freq="h"
    )
    if len(expected_index.difference(pd.DatetimeIndex(hourly["valid_time"]))):
        raise ValueError("ERA5 data contains missing UTC hours.")

    hourly["temperature_c"] = hourly["temperature_k"] - 273.15
    hourly["precipitation_mm"] = hourly["precipitation_m"] * 1000.0
    if (hourly["precipitation_mm"] < 0).any():
        raise ValueError("ERA5 data contains negative hourly precipitation.")
    hourly["time_utc"] = pd.to_datetime(hourly["valid_time"], utc=True)
    hourly["time_local"] = hourly["time_utc"].dt.tz_convert("Australia/Melbourne")
    return hourly


def aggregate_daily_weather(hourly):
    hourly = hourly.copy()
    hourly["temperature_date"] = (
        hourly["time_local"].dt.tz_localize(None).dt.normalize()
    )
    # ERA5 precipitation covers the preceding interval, so local midnight belongs to the prior day.
    precipitation_time = hourly["time_local"] - pd.Timedelta(nanoseconds=1)
    hourly["precipitation_date"] = (
        precipitation_time.dt.tz_localize(None).dt.normalize()
    )

    temperature = (
        hourly.groupby("temperature_date")
        .agg(
            temp_mean_c=("temperature_c", "mean"),
            temp_max_c=("temperature_c", "max"),
            temp_min_c=("temperature_c", "min"),
            temperature_hour_count=("temperature_c", "count"),
        )
        .reset_index()
        .rename(columns={"temperature_date": "date"})
    )
    precipitation = (
        hourly.groupby("precipitation_date")
        .agg(
            precipitation_mm=("precipitation_mm", "sum"),
            precipitation_hour_count=("precipitation_mm", "count"),
        )
        .reset_index()
        .rename(columns={"precipitation_date": "date"})
    )
    daily = pd.merge(temperature, precipitation, on="date", how="outer", validate="one_to_one")
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.loc[
        (daily["date"] >= "2010-01-01") & (daily["date"] < "2020-01-01")
    ].copy()
    complete = daily["temperature_hour_count"].between(23, 25) & daily[
        "precipitation_hour_count"
    ].between(23, 25)
    daily = daily.loc[complete].sort_values("date").reset_index(drop=True)
    if daily.isna().any().any():
        raise ValueError("Daily ERA5 data contains missing values.")
    return daily


def merge_datasets(supply, weather):
    weather_columns = ["temp_mean_c", "temp_max_c", "temp_min_c", "precipitation_mm"]
    merged = pd.merge(
        supply,
        weather[["date", *weather_columns]],
        on="date",
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if (merged["_merge"] != "both").any():
        raise ValueError("Some supply dates could not be matched to ERA5 data.")
    merged = merged.drop(columns="_merge")
    if merged[weather_columns].isna().any().any():
        raise ValueError("Merged weather features contain missing values.")
    return merged


def build_model_features(data):
    data = data.sort_values("date").reset_index(drop=True).copy()
    data["temp_range_c"] = data["temp_max_c"] - data["temp_min_c"]
    if (data["temp_range_c"] < 0).any():
        raise ValueError("Negative daily temperature range found.")
    data["precipitation_log1p"] = np.log1p(data["precipitation_mm"])
    data["annual_sin"] = np.sin(2.0 * np.pi * data["day_of_year"] / 365.25)
    data["annual_cos"] = np.cos(2.0 * np.pi * data["day_of_year"] / 365.25)
    data["demand_lag_1"] = data["supplied_ML"].shift(1)
    data["demand_lag_7"] = data["supplied_ML"].shift(7)
    prior_demand = data["supplied_ML"].shift(1)
    data["demand_rolling_mean_7"] = prior_demand.rolling(7, min_periods=7).mean()
    prior_precipitation = data["precipitation_mm"].shift(1)
    data["precipitation_previous_7d_sum"] = prior_precipitation.rolling(
        7, min_periods=7
    ).sum()

    required = [
        "supplied_ML",
        "is_public_holiday",
        "day_of_week",
        "temp_mean_c",
        "temp_range_c",
        "precipitation_log1p",
        "annual_sin",
        "annual_cos",
        "demand_lag_1",
        "demand_lag_7",
        "demand_rolling_mean_7",
        "precipitation_previous_7d_sum",
    ]
    data["model_ready"] = data[required].notna().all(axis=1)
    scored = data.loc[data["data_role"].isin(["development", "test"])]
    if not scored["model_ready"].all():
        raise ValueError("Development or test rows contain incomplete model features.")
    if data["date"].min() != DATA_START or data["date"].max() != DATA_END:
        raise ValueError("Prepared data has an unexpected date range.")
    return data


def main():
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    supply = add_calendar_features(load_supply_data(SUPPLY_FILE))
    hourly_weather = load_hourly_weather(ERA5_DIR)
    daily_weather = aggregate_daily_weather(hourly_weather)
    features = build_model_features(merge_datasets(supply, daily_weather))

    daily_weather.to_csv(WEATHER_OUTPUT, index=False, date_format="%Y-%m-%d")
    features.to_csv(FEATURE_OUTPUT, index=False, date_format="%Y-%m-%d")

    print(f"Prepared {len(features)} daily records from {DATA_START.date()} to {DATA_END.date()}.")
    print(f"Public holidays identified: {int(features['is_public_holiday'].sum())}")
    print(f"Daily ERA5 data saved to {WEATHER_OUTPUT}")
    print(f"Model features saved to {FEATURE_OUTPUT}")


if __name__ == "__main__":
    main()
