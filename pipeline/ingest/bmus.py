import os
import logging
from datetime import datetime

import requests
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# API endpoint
BMUS_URL = "https://data.elexon.co.uk/bmrs/api/v1/reference/bmunits/all"

# Output path
OUTPUT_DIR = os.path.join("data", "raw", "bmus")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "bmUnits.parquet")


def fetch_bmus_data() -> pd.DataFrame:
    """
    Fetch BM Units data from Elexon API.

    Returns
    -------
    pd.DataFrame
        DataFrame containing all BM Units data.
    """
    try:
        logger.info("Fetching BM Units data from API...")
        response = requests.get(BMUS_URL, timeout=30)
        response.raise_for_status()

        data = response.json()
        logger.info(f"Retrieved {len(data)} records from API.")

        df = pd.DataFrame(data)
        return df

    except requests.exceptions.RequestException as e:
        logger.error(f"Network error while fetching BM Units data: {e}")
        raise
    except ValueError as e:
        logger.error(f"Error decoding JSON response: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in fetch_bmus_data: {e}")
        raise


def save_bmus_data(df: pd.DataFrame) -> None:
    """
    Save BM Units data to a Parquet file, overwriting existing file.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to save.
    """
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        df.to_parquet(OUTPUT_FILE, index=False)
        logger.info(f"Saved BM Units data to {OUTPUT_FILE}")
    except Exception as e:
        logger.error(f"Error saving BM Units data: {e}")
        raise


def main():
    """Run BM Units data ingestion."""
    logger.info("Starting BM Units ingestion pipeline...")

    df = fetch_bmus_data()
    save_bmus_data(df)

    logger.info("BM Units ingestion pipeline completed successfully.")


if __name__ == "__main__":
    main()