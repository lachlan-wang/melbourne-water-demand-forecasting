# Data

Raw data are not committed to this repository. Create the paths below before
running the analysis, or use the optional ERA5 download script.

## Water supply

Place the source file at:

```text
data/raw/melbourne_water_daily_supply.csv
```

The required columns are:

- `recorddate`: observation timestamp in `%m/%d/%Y %I:%M:%S %p` format;
- `supplied_ML`: daily volume supplied, in megalitres;
- `ObjectId`: source identifier, retained in the raw file but not used by the model.

The source is the Melbourne Water Corporation dataset
[Water Supply - Daily Volume Drawn from storage dams operated by Melbourne Water](https://discover.data.vic.gov.au/dataset/water-supply-daily-volume-drawn-from-storage-dams-operated-by-melbourne-water/resource/2f386033-f8c3-4e47-b206-29d433cc7e24).
It reports accumulated volume supplied from ten Melbourne Water storages over
24-hour periods. The dataset is licensed under CC BY 4.0. This project uses
2010-11-01 to 2019-12-31.

## ERA5 weather

The analysis uses hourly ERA5 single-level reanalysis data.
Required variables are `2m_temperature` and `total_precipitation` for the area
`[37.5°S, 144.75°E, 38.0°S, 145.25°E]` and years 2010-2019.

Place the extracted NetCDF files under year folders:

```text
data/raw/era5/2010/*.nc
...
data/raw/era5/2019/*.nc
```

`scripts/download_era5.py` can obtain these files from the Copernicus Climate
Data Store after the user configures CDS API credentials. The download is
optional and is independent from model fitting.

The preparation script spatially averages the ERA5 grid, converts temperature
from kelvin to degrees Celsius and precipitation from metres to millimetres,
converts UTC timestamps to `Australia/Melbourne`, and aggregates hourly values
to local calendar days. Temperature is summarised by daily mean, maximum and
minimum; precipitation is summed.

## Calendar features

Victorian public holidays for 2010-2019 are generated with the Python
`holidays` package, including observed holidays. The script also creates day of
week, weekend, month and day-of-year fields. These calendar rules should be
treated as a reproducible software-derived calendar, not as an independently
audited legal holiday register.

## Generated files

`scripts/01_prepare_data.py` writes:

```text
data/processed/era5_daily.csv
data/processed/model_features.csv
```

Both files are reproducible intermediates and are ignored by Git.
