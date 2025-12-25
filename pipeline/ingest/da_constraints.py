#!/usr/bin/env python3
"""
da_constraints.py

Fetch NESO Day-Ahead Constraints data (FULL HISTORICAL DATASET).
- Downloads complete CSV from NESO API
- No date filtering (keeps all available data)
- Use this for INITIAL SETUP or full refresh

For daily incremental updates, use extract_incremental.py instead.

Run:
    python -m pipeline.ingest.da_constraints
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

DATE_COL = "Date (GMT/BST)"
OUTPUT_DIR = "data/raw/da_constraints"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "da_constraints.parquet")

# --------------------------------
# Logging setup
# --------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def fetch_da_constraints_full() -> pd.DataFrame:
    """
    Fetch complete Day-Ahead Constraints dataset from NESO.
    
    Downloads entire historical dataset (no date filtering).
    This is intended for initial setup or full refresh.

    Returns:
        pd.DataFrame: All constraints data from NESO

    Raises:
        RuntimeError: If download or CSV parsing fails
        ValueError: If expected date column is not found
    """
    logging.info("=" * 60)
    logging.info("DAY-AHEAD CONSTRAINTS - FULL DOWNLOAD")
    logging.info("=" * 60)
    logging.info("⚠️  This will download the ENTIRE historical dataset")
    logging.info("⚠️  For daily updates, use extract_incremental.py instead")
    logging.info("")

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Download CSV from NESO
    try:
        logging.info("Downloading from NESO API...")
        logging.info(f"URL: {NESO_URL}")
        response = requests.get(NESO_URL, timeout=60)  # Increased timeout for large file
        response.raise_for_status()
        logging.info("✅ Successfully downloaded data")
        logging.info(f"   Response size: {len(response.content) / (1024*1024):.2f} MB")
    except requests.RequestException as e:
        logging.error(f"❌ Failed to download: {e}")
        raise RuntimeError(f"Error downloading NESO data: {e}")

    # Parse CSV
    try:
        df = pd.read_csv(StringIO(response.text))
        logging.info(f"✅ Loaded {len(df):,} rows from CSV")
        logging.info(f"   Columns: {list(df.columns)}")
    except Exception as e:
        logging.error(f"❌ Failed to parse CSV: {e}")
        raise RuntimeError(f"Error parsing CSV data: {e}")

    # Validate date column
    if DATE_COL not in df.columns:
        logging.error(f"❌ Expected column '{DATE_COL}' not found")
        logging.info(f"Available columns: {list(df.columns)}")
        raise ValueError(f"Expected column '{DATE_COL}' not found in dataset")

    # Convert dates and handle invalid entries
    logging.info("\nProcessing dates...")
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    
    # Count and log invalid dates
    invalid_dates = df[DATE_COL].isna().sum()
    if invalid_dates > 0:
        logging.warning(f"⚠️  Found {invalid_dates} invalid dates (will be removed)")
    
    # Remove rows with invalid dates
    df = df.dropna(subset=[DATE_COL])
    
    # Log date range
    valid_dates = df[DATE_COL]
    if len(valid_dates) > 0:
        logging.info(f"✅ Date range: {valid_dates.min().date()} to {valid_dates.max().date()}")
        logging.info(f"   Total days covered: {valid_dates.nunique()}")
        logging.info(f"   Rows per day (avg): {len(df) / valid_dates.nunique():.1f}")
    
    # Log constraint groups
    if 'Constraint Group' in df.columns:
        constraint_groups = df['Constraint Group'].nunique()
        logging.info(f"   Constraint groups: {constraint_groups}")

    # Save to parquet
    df.to_parquet(OUTPUT_FILE, index=False)
    file_size_mb = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)
    logging.info(f"\n💾 Saved to {OUTPUT_FILE}")
    logging.info(f"   File size: {file_size_mb:.2f} MB")
    logging.info(f"   Total rows: {len(df):,}")
    
    logging.info("\n" + "=" * 60)
    logging.info("✅ FULL DOWNLOAD COMPLETED SUCCESSFULLY")
    logging.info("=" * 60)

    return df


if __name__ == "__main__":
    try:
        df = fetch_da_constraints_full()
        logging.info(f"\n🎉 Successfully downloaded {len(df):,} rows of constraints data")
    except Exception as e:
        logging.error(f"\n❌ Download failed: {e}")
        raise