import os
import glob
import logging
from datetime import datetime, time, timedelta
from typing import List, Optional

import requests
import pandas as pd
from time import sleep

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# API endpoint
BOALF_URL = "https://data.elexon.co.uk/bmrs/api/v1/balancing/acceptances"

# Input and output paths
BMUS_FILE = os.path.join("data", "raw", "bmus", "bmUnits.parquet")
OUTPUT_DIR = os.path.join("data", "raw", "boalf")
FINAL_OUTPUT_FILE = os.path.join(OUTPUT_DIR, "boalf.parquet")

# Fixed date range
DATE_FROM = datetime(2024, 1, 1, 0, 0)
yesterday = datetime.today().date() - timedelta(days=1)
DATE_TO = datetime.combine(yesterday, time(23, 30))

# Retry configuration
MAX_RETRIES = 3
BACKOFF_FACTOR = 2  # exponential backoff


def fetch_boalf_chunk(bm_unit: str, start: datetime, end: datetime) -> Optional[pd.DataFrame]:
    """
    Fetch BOALF data for a single BM Unit over a time chunk.
    Retries on failure with exponential backoff.
    """
    params = {
        "bmUnit": bm_unit,
        "from": start.strftime("%Y-%m-%dT%H:%MZ"),
        "to": end.strftime("%Y-%m-%dT%H:%MZ"),
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(BOALF_URL, params=params, timeout=60)
            response.raise_for_status()
            data = response.json().get("data", [])

            if not data:
                return pd.DataFrame()

            return pd.DataFrame(data)

        except requests.exceptions.RequestException as e:
            wait_time = BACKOFF_FACTOR ** (attempt - 1)
            logger.warning(
                f"Attempt {attempt}/{MAX_RETRIES} failed for {bm_unit} "
                f"[{start} → {end}]: {e}. Retrying in {wait_time}s..."
            )
            sleep(wait_time)

        except ValueError as e:
            logger.error(f"JSON decode error for {bm_unit} [{start} → {end}]: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error for {bm_unit} [{start} → {end}]: {e}")
            return None

    logger.error(f"All {MAX_RETRIES} attempts failed for {bm_unit} [{start} → {end}]")
    return None


def fetch_boalf_for_bmunit(bm_unit: str) -> pd.DataFrame:
    """
    Fetch BOALF data for a BM Unit in 7-day chunks and combine.
    """
    chunk_size = timedelta(days=7)
    all_chunks: List[pd.DataFrame] = []

    current_start = DATE_FROM
    while current_start < DATE_TO:
        current_end = min(current_start + chunk_size, DATE_TO)
        df = fetch_boalf_chunk(bm_unit, current_start, current_end)
        if df is not None and not df.empty:
            all_chunks.append(df)
        current_start = current_end + timedelta(minutes=30)  # step forward

    if not all_chunks:
        logger.warning(f"No BOALF data collected for {bm_unit}")
        return pd.DataFrame()

    return pd.concat(all_chunks, ignore_index=True)


def save_intermediate(df: pd.DataFrame, bm_unit: str) -> str:
    """
    Save intermediate BOALF data for a BM Unit.
    """
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        file_path = os.path.join(OUTPUT_DIR, f"{bm_unit}.parquet")
        df.to_parquet(file_path, index=False)
        logger.info(f"Saved intermediate BOALF data for {bm_unit} -> {file_path}")
        return file_path
    except Exception as e:
        logger.error(f"Error saving intermediate file for {bm_unit}: {e}")
        return ""


def merge_intermediates(final_path: str) -> None:
    """
    Merge all per-BMU parquet files into a single boalf.parquet,
    then delete intermediates.
    """
    try:
        files = glob.glob(os.path.join(OUTPUT_DIR, "*.parquet"))
        files = [f for f in files if not f.endswith("boalf.parquet")]  # exclude final

        if not files:
            logger.warning("No intermediate files found to merge.")
            return

        logger.info(f"Merging {len(files)} BMU files into {final_path}...")
        dfs = [pd.read_parquet(f) for f in files if os.path.getsize(f) > 0]
        if not dfs:
            logger.warning("No non-empty BMU files found.")
            return

        merged_df = pd.concat(dfs, ignore_index=True)
        merged_df.to_parquet(final_path, index=False)
        logger.info(f"Final BOALF data saved to {final_path}")

        # Clean up intermediates
        for f in files:
            os.remove(f)
        logger.info("Intermediate files deleted.")

    except Exception as e:
        logger.error(f"Error merging BMU files: {e}")
        raise


def main():
    """Run BOALF ingestion pipeline."""
    logger.info("Starting BOALF ingestion pipeline...")

    # Load BM Units and filter to WIND, excluding 'None'
    if not os.path.exists(BMUS_FILE):
        logger.error(f"BM Units file not found: {BMUS_FILE}")
        return

    bmus_df = pd.read_parquet(BMUS_FILE)
    wind_bmus = (
        bmus_df[bmus_df["fuelType"] == "WIND"]["elexonBmUnit"]
        .dropna()
        .loc[lambda x: x != "None"]
        .unique()
    )

    logger.info(f"Found {len(wind_bmus)} valid WIND BM Units to fetch.")

    # Fetch and save intermediates with progress counter
    total = len(wind_bmus)
    for idx, bm_unit in enumerate(wind_bmus, start=1):
        logger.info(f"[{idx}/{total}] Fetching BOALF for BMU: {bm_unit}")
        df = fetch_boalf_for_bmunit(bm_unit)
        if not df.empty:
            save_intermediate(df, bm_unit)

    # Merge intermediates into final file
    merge_intermediates(FINAL_OUTPUT_FILE)

    logger.info("BOALF ingestion pipeline completed successfully.")


if __name__ == "__main__":
    main()