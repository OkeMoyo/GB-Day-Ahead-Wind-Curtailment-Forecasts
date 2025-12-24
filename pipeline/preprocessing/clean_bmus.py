# clean_bmus.py
"""
Data cleaning script for BMUs dataset.
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
RAW_PATH = os.path.join("data", "raw", "bmus", "bmUnits.parquet")
PROCESSED_DIR = os.path.join("data", "processed", "bmus")
os.makedirs(PROCESSED_DIR, exist_ok=True)
OUTPUT_PATH = os.path.join(PROCESSED_DIR, "bmus_cleaned.parquet")

def clean_bmus():
    """Load, clean, and save BMUs dataset."""
    logging.info("Starting BMUs data cleaning...")

    # Load raw data
    try:
        bmus = pd.read_parquet(RAW_PATH)
        logging.info(f"Loaded BMUs data with shape {bmus.shape}")
    except Exception as e:
        logging.error(f"Failed to load BMUs data from {RAW_PATH}: {e}")
        raise

    # Drop rows with missing or 'None' BM Unit IDs
    bmus = bmus.dropna(subset=['elexonBmUnit'])
    bmus = bmus[bmus['elexonBmUnit'] != 'None']

    # Filter for wind fuel type only
    bmus = bmus[bmus['fuelType'] == 'WIND']
    logging.info(f"Filtered for wind BMUs. Remaining rows: {len(bmus)}")

    # Drop unnecessary columns
    drop_cols = [
        'nationalGridBmUnit', 'eic', 'fuelType', 'bmUnitType', 'fpnFlag',
        'leadPartyId', 'productionOrConsumptionFlag', 'demandCapacity',
        'transmissionLossFactor', 'workingDayCreditAssessmentImportCapability',
        'nonWorkingDayCreditAssessmentImportCapability',
        'workingDayCreditAssessmentExportCapability',
        'nonWorkingDayCreditAssessmentExportCapability',
        'creditQualifyingStatus', 'demandInProductionFlag', 'gspGroupId',
        'gspGroupName', 'interconnectorId'
    ]
    bmus_cleaned = bmus.drop(columns=drop_cols, errors='ignore')

    # Rename columns
    bmus_cleaned = bmus_cleaned.rename(columns={'elexonBmUnit': 'bmUnit'})

    # Update BMU names for specific IDs
    update_dict = {
        'T_ABRBO-1': 'Aberdeen Offshore Wind Farm',
        'T_GYMR-15': 'Gwynt y Mor Offshore WF',
        'T_GYMR-17': 'Gwynt y Mor Offshore WF',
        'T_GYMR-26': 'Gwynt y Mor Offshore WF',
        'T_GYMR-28': 'Gwynt y Mor Offshore WF',
        'T_PNYCW-1': 'Pen y Cymoedd Wind Farm Ltd.',
        'T_RCBKO-1': 'Race Bank Wind Farm',
        'T_RCBKO-2': 'Race Bank Wind Farm',
    }

    for bmunit, new_name in update_dict.items():
        bmus_cleaned.loc[bmus_cleaned['bmUnit'] == bmunit, 'bmUnitName'] = new_name

    # Save cleaned data
    try:
        bmus_cleaned.to_parquet(OUTPUT_PATH, index=False)
        logging.info(f"Saved cleaned BMUs data to {OUTPUT_PATH}")
    except Exception as e:
        logging.error(f"Failed to save cleaned BMUs data: {e}")
        raise

    logging.info("BMUs data cleaning completed successfully.")
    return bmus_cleaned

if __name__ == "__main__":
    clean_bmus()