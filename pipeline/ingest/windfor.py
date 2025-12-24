import os
import time
import json
import logging
import requests
import pandas as pd
from datetime import datetime

# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------
DATA_DIR = "data/raw/windfor"
OUTPUT_FILE = "data/raw/windfor/windfor.parquet"

# Configurable date range
START_DATE = "2024-01-01"
END_DATE = datetime.now().strftime("%Y-%m-%d")

API_URL = "https://data.elexon.co.uk/bmrs/api/v1/forecast/generation/wind/latest/stream"
MAX_RETRIES = 3
BACKOFF_FACTOR = 1  # seconds

# -------------------------------------------------------------------
# Logging setup
# -------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

# -------------------------------------------------------------------
# Functions
# -------------------------------------------------------------------
def fetch_wind_forecast(start_date, end_date):
    """Fetch wind generation forecast data with retry logic."""
    params = {
        "from": start_date,
        "to": end_date
    }

    retries = 0
    while retries < MAX_RETRIES:
        try:
            response = requests.get(API_URL, params=params, headers={"accept": "application/json"}, timeout=60)
            response.raise_for_status()
            data = response.json()

            # The API returns a JSON array directly (not wrapped in "data")
            if isinstance(data, dict) and "data" in data:
                df = pd.DataFrame(data.get("data", []))
            else:
                df = pd.DataFrame(data)

            return df
        except Exception as e:
            retries += 1
            wait_time = BACKOFF_FACTOR * (2 ** (retries - 1))
            logging.warning(
                f"Retry {retries}/{MAX_RETRIES} for wind forecast due to error: {e}"
            )
            time.sleep(wait_time)

    logging.error("Failed to fetch wind forecast data after maximum retries.")
    return pd.DataFrame()


def save_wind_forecast(df, output_path):
    """Save wind forecast dataframe to Parquet."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, index=False)
    logging.info(f"Saved wind forecast data to {output_path}")


def run_extraction():
    """Main extraction pipeline for wind forecast data."""
    logging.info(f"Fetching wind forecast data from {START_DATE} to {END_DATE}")
    df = fetch_wind_forecast(START_DATE, END_DATE)

    if df.empty:
        logging.warning("No wind forecast data retrieved.")
        return

    save_wind_forecast(df, OUTPUT_FILE)


# -------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------
if __name__ == "__main__":
    run_extraction()