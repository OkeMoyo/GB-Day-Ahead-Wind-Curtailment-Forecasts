#!/usr/bin/env python3
"""
da_constraints.py

Fetch NESO Day-Ahead Constraints data as a DataFrame.
- Download CSV from NESO API
- Filter by provided start/end dates
- Returns a pandas DataFrame
- Defaults to 2024-01-01 → TODAY
"""

import os
import logging
import requests
import pandas as pd
from datetime import datetime
from io import StringIO

# --------------------------------
# Configuration
# --------------------------------
NESO_URL = (
    "https://api.neso.energy/dataset/cf3cbc92-2d5d-4c2b-bd29-e11a21070b26/"
    "resource/38a18ec1-9e40-465d-93fb-301e80fd1352/download/"
    "day-ahead-constraints-limits-and-flow-output-v1.5.csv"
)

# Default start/end dates and expected date column name
START_DATE = "2024-01-01"
# Use today's date as end date (dynamically updated each run)
END_DATE = datetime.now().strftime("%Y-%m-%d")
DATE_COL = "Date (GMT/BST)"
OUTPUT_DIR = "data/raw/da_constraints"

# --------------------------------
# Logging setup
# --------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def fetch_da_constraints_data(start: datetime | None = None, end: datetime | None = None) -> pd.DataFrame:
    """
    Fetch Day-Ahead Constraints data between start and end dates.

    Args:
        start (datetime | None): Start datetime (inclusive). Defaults to 2024-01-01.
        end (datetime | None): End datetime (inclusive). Defaults to today.

    Returns:
        pd.DataFrame: Filtered constraints data

    Raises:
        RuntimeError: If download or CSV parsing fails
        ValueError: If expected date column is not found
    """
    # Use defaults if start/end not provided
    if start is None:
        start = pd.to_datetime(START_DATE)
    if end is None:
        end = pd.to_datetime(END_DATE)  # Uses today's date from module constant

    logging.info(f"Fetching Day-Ahead Constraints data from {start.date()} to {end.date()}")

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Download CSV from NESO
    try:
        logging.info("Downloading Day-Ahead Constraints data from NESO API...")
        response = requests.get(NESO_URL, timeout=30)
        response.raise_for_status()
        logging.info("✅ Successfully downloaded data from NESO API")
    except requests.RequestException as e:
        logging.error(f"Failed to download NESO Day-Ahead Constraints data: {e}")
        raise RuntimeError(f"Error downloading NESO Day-Ahead Constraints data: {e}")

    # Load CSV into DataFrame
    try:
        df = pd.read_csv(StringIO(response.text))
        logging.info(f"Loaded {len(df):,} rows from NESO CSV")
        logging.info(f"Dataset columns: {list(df.columns)}")
    except Exception as e:
        logging.error(f"Failed to parse CSV data: {e}")
        raise RuntimeError(f"Error reading NESO CSV data: {e}")

    # Validate expected date column exists
    if DATE_COL not in df.columns:
        logging.error(f"Expected date column '{DATE_COL}' not found in dataset")
        logging.info(f"Available columns: {list(df.columns)}")
        raise ValueError(f"Expected date column '{DATE_COL}' not found in NESO Day-Ahead Constraints dataset")

    # Convert to datetime and validate
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    
    # Check for invalid dates
    invalid_dates = df[DATE_COL].isna().sum()
    if invalid_dates > 0:
        logging.warning(f"Found {invalid_dates} invalid dates in {DATE_COL} column")
    
    # Log date range in source data
    valid_dates = df[DATE_COL].dropna()
    if len(valid_dates) > 0:
        logging.info(f"Date range in source data: {valid_dates.min().date()} to {valid_dates.max().date()}")

    # Filter by requested range
    mask = (df[DATE_COL] >= pd.to_datetime(start)) & (df[DATE_COL] <= pd.to_datetime(end))
    filtered_df = df.loc[mask].copy()
    
    # Validate filtered results
    if len(filtered_df) == 0:
        logging.warning(f"No data found between {start.date()} and {end.date()}")
        logging.info("This may be expected if the data source hasn't been updated yet")
    else:
        filtered_date_range = filtered_df[DATE_COL].dropna()
        if len(filtered_date_range) > 0:
            logging.info(f"Filtered dataset to {len(filtered_df):,} rows")
            logging.info(f"Actual date range in filtered data: {filtered_date_range.min().date()} to {filtered_date_range.max().date()}")

    # Save to parquet file
    output_file = os.path.join(OUTPUT_DIR, "da_constraints.parquet")
    filtered_df.to_parquet(output_file, index=False)
    logging.info(f"💾 Saved data to {output_file}")

    return filtered_df


if __name__ == "__main__":
    logging.info("Starting Day-Ahead Constraints data fetch...")
    logging.info(f"Using date range: {START_DATE} to {END_DATE} (today)")
    try:
        df = fetch_da_constraints_data()
        logging.info(f"🎉 Successfully fetched {len(df):,} rows of Day-Ahead Constraints data")
    except Exception as e:
        logging.error(f"❌ Failed to fetch Day-Ahead Constraints data: {e}")
        raise