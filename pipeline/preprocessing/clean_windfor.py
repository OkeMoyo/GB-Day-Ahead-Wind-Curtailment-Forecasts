# clean_windfor.py
"""
Data cleaning script for national wind generation forecasts (WINDFOR).
Part of the wind curtailment modelling MLOps pipeline.
"""

import os
import pandas as pd
import logging

# -------------------------------
# Logging configuration
# -------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

# -------------------------------
# File paths
# -------------------------------
RAW_PATH = os.path.join("data", "raw", "windfor", "windfor.parquet")
PROCESSED_DIR = os.path.join("data", "processed", "windfor")
os.makedirs(PROCESSED_DIR, exist_ok=True)
OUTPUT_PATH = os.path.join(PROCESSED_DIR, "windfor_cleaned.parquet")

def clean_windfor():
    """Load, clean, and save national wind forecast dataset."""
    logging.info("Starting WINDFOR data cleaning...")

    # Load raw data
    try:
        windfor = pd.read_parquet(RAW_PATH)
        logging.info(f"Loaded WINDFOR data with shape {windfor.shape}")
    except Exception as e:
        logging.error(f"Failed to load WINDFOR data from {RAW_PATH}: {e}")
        raise

    # Drop unnecessary columns
    windfor_cleaned = windfor.drop(
        columns=['publishTime', 'settlementDate', 'settlementPeriod'],
        errors='ignore'
    )

    # Rename columns for clarity
    windfor_cleaned = windfor_cleaned.rename(
        columns={
            'startTime': 'time',
            'generation': 'sys_wind_gen_forecast'
        }
    )

    # Convert time column to datetime
    windfor_cleaned['time'] = pd.to_datetime(windfor_cleaned['time'])

    # Convert to half-hourly resolution
    windfor_half_hourly = []
    for idx, row in windfor_cleaned.iterrows():
        for offset in [0, 30]:
            windfor_half_hourly.append({
                'half_hour_time': row['time'] + pd.Timedelta(minutes=offset),
                'sys_wind_gen_forecast': row['sys_wind_gen_forecast']
            })

    windfor_cleaned = pd.DataFrame(windfor_half_hourly)

    # Ensure timezone-naive datetime for merging consistency
    windfor_cleaned['half_hour_time'] = pd.to_datetime(
        windfor_cleaned['half_hour_time']
    ).dt.tz_localize(None)

    logging.info(f"Transformed to half-hourly resolution. New shape: {windfor_cleaned.shape}")

    # Save cleaned data
    try:
        windfor_cleaned.to_parquet(OUTPUT_PATH, index=False)
        logging.info(f"Saved cleaned WINDFOR data to {OUTPUT_PATH}")
    except Exception as e:
        logging.error(f"Failed to save cleaned WINDFOR data: {e}")
        raise

    logging.info("WINDFOR data cleaning completed successfully.")
    return windfor_cleaned

if __name__ == "__main__":
    clean_windfor()