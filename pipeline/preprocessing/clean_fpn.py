# clean_fpn.py
"""
Data cleaning script for FPN dataset.
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
RAW_PATH = os.path.join("data", "raw", "fpn", "fpn.parquet")
PROCESSED_DIR = os.path.join("data", "processed", "fpn")
os.makedirs(PROCESSED_DIR, exist_ok=True)
OUTPUT_PATH = os.path.join(PROCESSED_DIR, "fpn_cleaned.parquet")

def clean_fpn():
    """Load, clean, and save FPN dataset."""
    logging.info("Starting FPN data cleaning...")

    # Load raw data
    try:
        fpn = pd.read_parquet(RAW_PATH)
        logging.info(f"Loaded FPN data with shape {fpn.shape}")
    except Exception as e:
        logging.error(f"Failed to load FPN data from {RAW_PATH}: {e}")
        raise

    # Drop unnecessary columns
    drop_cols = ['nationalGridBmUnit', 'dataset']
    fpn_cleaned = fpn.drop(columns=drop_cols, errors='ignore')

    # Sort data for consistency
    fpn_cleaned = fpn_cleaned.sort_values(
        by=['bmUnit', 'settlementDate', 'settlementPeriod']
    ).reset_index(drop=True)

    # Rename columns for clarity
    fpn_cleaned = fpn_cleaned.rename(
        columns={
            'levelFrom': 'fpn_levelFrom',
            'levelTo': 'fpn_levelTo'
        }
    )

    # Drop duplicates
    before = len(fpn_cleaned)
    fpn_cleaned = fpn_cleaned.drop_duplicates()
    after = len(fpn_cleaned)
    logging.info(f"Dropped {before - after} duplicate rows from FPN data")

    # Save cleaned data
    try:
        fpn_cleaned.to_parquet(OUTPUT_PATH, index=False)
        logging.info(f"Saved cleaned FPN data to {OUTPUT_PATH}")
    except Exception as e:
        logging.error(f"Failed to save cleaned FPN data: {e}")
        raise

    logging.info("FPN data cleaning completed successfully.")
    return fpn_cleaned

if __name__ == "__main__":
    clean_fpn()