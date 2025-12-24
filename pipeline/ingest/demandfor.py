import os
import time
import logging
import requests
import pandas as pd
from datetime import datetime

# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------
OUTPUT_DIR = "data/raw/demandfor"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "demandfor.parquet")

BASE_URL = "https://api.neso.energy/api/3/action/datastore_search"
RESOURCE_ID = "9847e7bb-986e-49be-8138-717b25933fbb"

# NESO API pagination
LIMIT = 1000

# Configurable date range
START_DATE = "2024-01-01"
END_DATE = datetime.now().strftime("%Y-%m-%d")

# Retry settings
MAX_RETRIES = 3
BACKOFF_FACTOR = 1  # faster retries, same as others

# -------------------------------------------------------------------
# Logging setup
# -------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# Functions
# -------------------------------------------------------------------
def fetch_demand_forecast():
    """
    Fetch all demand forecast data from the NESO API with pagination and retry logic.
    """
    all_records = []
    offset = 0

    while True:
        params = {
            "resource_id": RESOURCE_ID,
            "limit": LIMIT,
            "offset": offset
        }

        retries = 0
        while retries < MAX_RETRIES:
            try:
                response = requests.get(BASE_URL, params=params, timeout=60)
                response.raise_for_status()
                data = response.json()
                records = data["result"]["records"]
                total = data["result"]["total"]

                if not records:
                    logger.info("No more records to fetch.")
                    return all_records

                all_records.extend(records)
                logger.info(f"Fetched {len(records)} records (offset {offset})")

                offset += LIMIT
                if offset >= total:
                    logger.info("Reached total record count. Fetching complete.")
                    return all_records

                break  # exit retry loop if successful

            except Exception as e:
                retries += 1
                wait_time = BACKOFF_FACTOR * (2 ** (retries - 1))
                logger.warning(
                    f"Retry {retries}/{MAX_RETRIES} for offset {offset} due to error: {e}"
                )
                time.sleep(wait_time)


def filter_date_range(df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Filter dataframe to the specified date range based on whichever column contains 'date'.
    """
    date_col = None
    for col in df.columns:
        if "date" in col.lower():
            date_col = col
            break

    if date_col is None:
        logger.warning("No date column found for filtering. Returning full dataset.")
        return df

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    mask = (df[date_col] >= start_date) & (df[date_col] <= end_date)
    filtered_df = df.loc[mask].copy()
    logger.info(f"Filtered from {len(df)} to {len(filtered_df)} records based on {date_col}.")
    return filtered_df


def save_to_parquet(df: pd.DataFrame, output_file: str):
    """
    Save dataframe to Parquet, overwriting existing file.
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_parquet(output_file, index=False)
    logger.info(f"Saved demand forecast data to {output_file}")


def run_extraction():
    """Main extraction pipeline for demand forecast data."""
    logger.info("Starting demand forecast extraction...")

    records = fetch_demand_forecast()
    if not records:
        logger.warning("No records fetched from NESO API.")
        return

    df = pd.DataFrame(records)
    logger.info(f"Total records fetched: {len(df)}")

    df = filter_date_range(df, START_DATE, END_DATE)

    if df.empty:
        logger.warning("No records remain after date filtering.")
        return

    save_to_parquet(df, OUTPUT_FILE)
    logger.info("Demand forecast extraction completed successfully.")


# -------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------
if __name__ == "__main__":
    run_extraction()