from pathlib import Path
import zipfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "era5"

DATASET = "reanalysis-era5-single-levels"
YEARS = range(2010, 2020)
VARIABLES = ["2m_temperature", "total_precipitation"]
AREA = [-37.5, 144.75, -38.0, 145.25]  # North, west, south, east
MONTHS = [f"{month:02d}" for month in range(1, 13)]
DAYS = [f"{day:02d}" for day in range(1, 32)]
TIMES = [f"{hour:02d}:00" for hour in range(24)]


def zip_is_complete(path):
    if not path.exists() or path.stat().st_size == 0 or not zipfile.is_zipfile(path):
        return False
    with zipfile.ZipFile(path) as archive:
        return bool(archive.namelist()) and archive.testzip() is None


def extract_archive(zip_path, output_dir):
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.namelist():
            member_path = Path(member)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise ValueError(f"Unsafe path in archive: {member}")
        archive.extractall(output_dir)

    netcdf_files = sorted(output_dir.rglob("*.nc"))
    if not netcdf_files or any(path.stat().st_size == 0 for path in netcdf_files):
        raise RuntimeError(f"No complete NetCDF files found in {output_dir}")
    return netcdf_files


def download_year(client, year):
    year_dir = OUTPUT_DIR / str(year)
    existing_files = sorted(year_dir.glob("*.nc"))
    if existing_files and all(path.stat().st_size > 0 for path in existing_files):
        print(f"{year}: existing NetCDF files found; download skipped.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = OUTPUT_DIR / f"era5_melbourne_hourly_{year}.zip"
    partial_path = zip_path.with_suffix(".zip.part")

    if partial_path.exists():
        partial_path.unlink()

    request = {
        "product_type": ["reanalysis"],
        "variable": VARIABLES,
        "year": [str(year)],
        "month": MONTHS,
        "day": DAYS,
        "time": TIMES,
        "data_format": "netcdf",
        "download_format": "zip",
        "area": AREA,
    }

    print(f"{year}: submitting CDS request.")
    result = client.retrieve(DATASET, request)
    result.download(str(partial_path))
    if not zip_is_complete(partial_path):
        raise RuntimeError(f"Downloaded archive failed validation: {partial_path}")

    partial_path.replace(zip_path)
    year_dir.mkdir(parents=True, exist_ok=True)
    files = extract_archive(zip_path, year_dir)
    print(f"{year}: downloaded and extracted {len(files)} NetCDF files.")


def main():
    try:
        import cdsapi
    except ImportError as error:
        raise ImportError(
            "cdsapi is required for the optional download step. Install project requirements first."
        ) from error

    client = cdsapi.Client(timeout=600, quiet=False)
    for year in YEARS:
        download_year(client, year)
    print(f"ERA5 downloads are available in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
