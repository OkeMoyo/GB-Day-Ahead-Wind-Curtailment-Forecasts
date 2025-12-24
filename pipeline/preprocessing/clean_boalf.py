# clean_boalf.py
"""
Data cleaning script for BOALF dataset.
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
RAW_PATH = os.path.join("data", "raw", "boalf", "boalf.parquet")
PROCESSED_DIR = os.path.join("data", "processed", "boalf")
os.makedirs(PROCESSED_DIR, exist_ok=True)
OUTPUT_PATH = os.path.join(PROCESSED_DIR, "boalf_cleaned.parquet")

def clean_boalf():
    """Load, clean, and save BOALF dataset."""
    logging.info("Starting BOALF data cleaning...")

    # Load raw data
    try:
        boalf = pd.read_parquet(RAW_PATH)
        logging.info(f"Loaded BOALF data with shape {boalf.shape}")
    except Exception as e:
        logging.error(f"Failed to load BOALF data from {RAW_PATH}: {e}")
        raise

    # Drop unnecessary columns
    drop_cols = [
        'nationalGridBmUnit', 'acceptanceNumber', 'acceptanceTime',
        'deemedBoFlag', 'storFlag', 'rrFlag'
    ]
    boalf_cleaned = boalf.drop(columns=drop_cols, errors='ignore')

    # Sort data for consistency
    boalf_cleaned = boalf_cleaned.sort_values(
        by=['settlementDate', 'settlementPeriodFrom', 'bmUnit']
    ).reset_index(drop=True)

    # Rename columns
    boalf_cleaned = boalf_cleaned.rename(
        columns={
            'levelFrom': 'boal_levelFrom',
            'levelTo': 'boal_levelTo'
        }
    )

    # Drop duplicates
    before = len(boalf_cleaned)
    boalf_cleaned = boalf_cleaned.drop_duplicates()
    after = len(boalf_cleaned)
    logging.info(f"Dropped {before - after} duplicate rows from BOALF data")

    # Save cleaned data
    try:
        boalf_cleaned.to_parquet(OUTPUT_PATH, index=False)
        logging.info(f"Saved cleaned BOALF data to {OUTPUT_PATH}")
    except Exception as e:
        logging.error(f"Failed to save cleaned BOALF data: {e}")
        raise

    logging.info("BOALF data cleaning completed successfully.")
    return boalf_cleaned

if __name__ == "__main__":
    clean_boalf()