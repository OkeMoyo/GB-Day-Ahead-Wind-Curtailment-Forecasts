"""
Unified incremental extraction pipeline (Parquet version).

This script loops through supported datasets (excluding BMUs), checks
the last date in the existing local Parquet file, queries the API only
for new data, saves it to intermediate files, then appends to final files.

Run:
    python -m pipeline.ingest.extract_incremental
"""

import os
import sys
import glob
import logging
import traceback
import pandas as pd
from datetime import datetime, timedelta, timezone

# Make sure imports work when running as a module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.ingest.boalf import fetch_boalf_chunk
from pipeline.ingest.fpn import fetch_fpn_for_bmu
from pipeline.ingest.demandfor import fetch_demand_forecast
from pipeline.ingest.windfor import fetch_wind_forecast
from pipeline.ingest.da_constraints import fetch_da_constraints_full

# ============================
# CONFIG
# ============================
RAW_DATA_DIR = os.path.join("data", "raw")
BMUS_FILE = os.path.join(RAW_DATA_DIR, "bmus", "bmUnits.parquet")

# Configurable: How many days back from today should we fetch up to?
API_LAG_DAYS = 0  # Can be changed to 2, 3, etc.

DATASETS = {
    "boalf": {"date_col": "settlementDate", "is_bmu_level": True},
    "fpn": {"date_col": "settlementDate", "is_bmu_level": True},
    "demandfor": {"date_col": "TARGETDATE", "is_bmu_level": False},
    "windfor": {"date_col": "settlementDate", "is_bmu_level": False},
    "da_constraints": {"date_col": "Date (GMT/BST)", "is_bmu_level": False},
}

CHUNK_SIZE_DAYS = 7  # For BMU-level extractions

# ============================
# LOGGING
# ============================
def setup_logging():
    """Setup logging with both console and file output."""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(os.path.join(log_dir, "incremental_extraction.log"), mode="a")
        ]
    )

# ============================
# HELPERS
# ============================
def get_last_available_date(file_path: str, date_col: str) -> datetime | None:
    """Return the max date in the existing local Parquet for the given dataset."""
    if not os.path.exists(file_path):
        logging.warning(f"No local file found: {file_path}. Full extraction needed.")
        return None
    
    try:
        df = pd.read_parquet(file_path, columns=[date_col])
        if df.empty:
            logging.warning(f"Empty dataset found: {file_path}")
            return None
        
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        valid_dates = df[date_col].dropna()
        if valid_dates.empty:
            logging.warning(f"No valid dates found in {file_path}")
            return None
            
        last_date = valid_dates.max()
        logging.info(f"Last available date in {os.path.basename(file_path)}: {last_date.date()}")
        return last_date
    except Exception as e:
        logging.error(f"Error reading {file_path}: {e}")
        return None


def get_api_latest_date() -> datetime:
    """Determine the latest date available on the API (today - API_LAG_DAYS)."""
    latest = datetime.now(timezone.utc) - timedelta(days=API_LAG_DAYS)
    return datetime.combine(latest.date(), datetime.min.time()).replace(tzinfo=None)


def save_intermediate(df: pd.DataFrame, dataset_name: str, identifier: str = "") -> str:
    """Save intermediate data to a parquet file."""
    intermediate_dir = os.path.join(RAW_DATA_DIR, dataset_name, "intermediate")
    os.makedirs(intermediate_dir, exist_ok=True)
    
    if identifier:
        file_path = os.path.join(intermediate_dir, f"{identifier}.parquet")
    else:
        file_path = os.path.join(intermediate_dir, "new_data.parquet")
    
    df.to_parquet(file_path, index=False)
    logging.debug(f"Saved intermediate data to {file_path}")
    return file_path


def merge_intermediates_to_final(dataset_name: str, final_path: str):
    """Merge all intermediate parquet files into the final dataset file."""
    intermediate_dir = os.path.join(RAW_DATA_DIR, dataset_name, "intermediate")
    
    if not os.path.exists(intermediate_dir):
        logging.warning(f"No intermediate directory found for {dataset_name}")
        return
    
    intermediate_files = glob.glob(os.path.join(intermediate_dir, "*.parquet"))
    
    if not intermediate_files:
        logging.info(f"No intermediate files to merge for {dataset_name}")
        return
    
    try:
        # Load all intermediate files
        new_data_dfs = [pd.read_parquet(f) for f in intermediate_files if os.path.getsize(f) > 0]
        
        if not new_data_dfs:
            logging.info(f"No non-empty intermediate files for {dataset_name}")
            return
        
        new_data = pd.concat(new_data_dfs, ignore_index=True)
        logging.info(f"Loaded {len(new_data):,} rows from {len(new_data_dfs)} intermediate files")
        
        # Append to existing final file
        if os.path.exists(final_path):
            existing_df = pd.read_parquet(final_path)
            logging.info(f"Existing final data: {len(existing_df):,} rows")
            combined_df = pd.concat([existing_df, new_data], ignore_index=True)
            logging.info(f"Combined data: {len(combined_df):,} rows")
        else:
            combined_df = new_data
            logging.info(f"Creating new final file with {len(combined_df):,} rows")
        
        # Save combined data
        os.makedirs(os.path.dirname(final_path), exist_ok=True)
        combined_df.to_parquet(final_path, index=False)
        logging.info(f"✅ Saved updated {os.path.basename(final_path)} with {len(combined_df):,} total rows")
        
        # Clean up intermediate files
        for f in intermediate_files:
            os.remove(f)
        os.rmdir(intermediate_dir)
        logging.info(f"Cleaned up {len(intermediate_files)} intermediate files")
        
    except Exception as e:
        logging.error(f"Error merging intermediate files for {dataset_name}: {e}")
        raise


def filter_data_by_date_range(df: pd.DataFrame, date_col: str, start_date: datetime) -> pd.DataFrame:
    """Filter DataFrame to only include records after start_date."""
    if df.empty:
        return df
    
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    mask = df[date_col] > start_date
    filtered_df = df.loc[mask].copy()
    
    logging.info(f"Filtered from {len(df)} to {len(filtered_df)} records after {start_date.date()}")
    return filtered_df


# ============================
# INCREMENTAL EXTRACTION FUNCTIONS
# ============================
def extract_bmu_level_incremental(dataset_name: str, date_col: str):
    """Incrementally extract BMU-level dataset (BOALF or FPN)."""
    logging.info(f"🔸 Starting BMU-level incremental extraction for {dataset_name}...")

    try:
        if not os.path.exists(BMUS_FILE):
            logging.error(f"❌ BMUs file not found: {BMUS_FILE}")
            return

        bmus_df = pd.read_parquet(BMUS_FILE)
        wind_bmus = (
            bmus_df[bmus_df["fuelType"] == "WIND"]["elexonBmUnit"]
            .dropna()
            .loc[lambda x: x != "None"]
            .unique()
        )

        logging.info(f"Found {len(wind_bmus)} WIND BMUs for {dataset_name} extraction.")

        file_path = os.path.join(RAW_DATA_DIR, dataset_name, f"{dataset_name}.parquet")
        last_date = get_last_available_date(file_path, date_col)
        latest_available_date = get_api_latest_date()

        if last_date is None:
            start_date = datetime(2024, 1, 1)
            logging.info(f"No existing data found. Starting from {start_date.date()}")
        else:
            start_date = last_date + timedelta(days=1)

        if start_date.date() > latest_available_date.date():
            logging.info(f"✅ No new data for {dataset_name}. Already up to date.")
            return

        logging.info(f"Fetching {dataset_name} data from {start_date.date()} to {latest_available_date.date()}")

        successful_fetches = 0
        failed_fetches = 0

        # Fetch data for each BMU
        for idx, bm_unit in enumerate(wind_bmus, start=1):
            try:
                logging.info(f"[{idx}/{len(wind_bmus)}] Fetching {dataset_name} for BMU {bm_unit}...")
                
                if dataset_name == "boalf":
                    # fetch_boalf_chunk takes datetime objects
                    all_chunks = []
                    current_start = start_date
                    
                    while current_start < latest_available_date:
                        current_end = min(current_start + timedelta(days=CHUNK_SIZE_DAYS), latest_available_date)
                        
                        df_chunk = fetch_boalf_chunk(
                            bm_unit=bm_unit,
                            start=current_start,
                            end=current_end
                        )
                        
                        if df_chunk is not None and not df_chunk.empty:
                            all_chunks.append(df_chunk)
                        
                        current_start = current_end + timedelta(minutes=30)
                    
                    if all_chunks:
                        bmu_data = pd.concat(all_chunks, ignore_index=True)
                        save_intermediate(bmu_data, dataset_name, bm_unit)
                        successful_fetches += 1
                        logging.info(f"✅ Fetched {len(bmu_data):,} rows for BMU {bm_unit}")
                    else:
                        logging.debug(f"No data returned for BMU {bm_unit}")
                        
                elif dataset_name == "fpn":
                    # fetch_fpn_for_bmu takes string dates and returns combined DataFrame
                    bmu_data = fetch_fpn_for_bmu(
                        bmu_id=bm_unit,  # ✅ Fixed: use bmu_id not bm_unit
                        start_date=start_date.strftime('%Y-%m-%dT%H:%MZ'),  # ✅ Fixed: string format
                        end_date=latest_available_date.strftime('%Y-%m-%dT%H:%MZ'),  # ✅ Fixed: string format
                        chunk_size_days=CHUNK_SIZE_DAYS
                    )
                    
                    if not bmu_data.empty:
                        save_intermediate(bmu_data, dataset_name, bm_unit)
                        successful_fetches += 1
                        logging.info(f"✅ Fetched {len(bmu_data):,} rows for BMU {bm_unit}")
                    else:
                        logging.debug(f"No data returned for BMU {bm_unit}")
                else:
                    logging.error(f"Unknown BMU-level dataset: {dataset_name}")
                    return
                    
            except Exception as e:
                failed_fetches += 1
                if "400" in str(e) or "Client Error" in str(e):
                    logging.warning(f"⚠️ BMU {bm_unit} returned 400 error - likely no data for date range")
                else:
                    logging.error(f"❌ Failed to fetch {dataset_name} for BMU {bm_unit}: {e}")
                continue

        logging.info(f"Fetch summary: {successful_fetches} successful, {failed_fetches} failed")

        if successful_fetches > 0:
            merge_intermediates_to_final(dataset_name, file_path)
            logging.info(f"✅ BMU-level incremental extraction complete for {dataset_name}")
        else:
            logging.warning(f"⚠️ No new {dataset_name} data retrieved from API")

    except Exception as e:
        logging.error(f"❌ Error in BMU-level extraction for {dataset_name}: {e}")
        logging.error(traceback.format_exc())
        raise  # ✅ Re-raise to be caught by main loop


def extract_system_level_incremental(dataset_name: str, date_col: str):
    """Incrementally extract system-level dataset (demandfor, windfor, da_constraints)."""
    logging.info(f"🔸 Starting system-level incremental extraction for {dataset_name}...")

    try:
        file_path = os.path.join(RAW_DATA_DIR, dataset_name, f"{dataset_name}.parquet")
        last_date = get_last_available_date(file_path, date_col)
        latest_available_date = get_api_latest_date()

        if last_date is None:
            logging.error(f"❌ No existing data found for {dataset_name}")
            logging.error(f"   Run initial extraction first: python -m pipeline.ingest.{dataset_name}")
            return

        start_date = last_date + timedelta(days=1)

        if start_date.date() > latest_available_date.date():
            logging.info(f"✅ No new data for {dataset_name}. Already up to date.")
            return

        logging.info(f"Fetching new {dataset_name} data from {start_date.date()} to {latest_available_date.date()}")
        
        # Handle different function signatures for different datasets
        if dataset_name == "demandfor":
            # demandfor fetches all data as a list of records, then we convert and filter
            all_records = fetch_demand_forecast()  # Returns list
            if not all_records:
                logging.info(f"ℹ️ No {dataset_name} data returned from API")
                return
            
            all_data = pd.DataFrame(all_records)
            new_data = filter_data_by_date_range(all_data, date_col, last_date)
            
        elif dataset_name == "da_constraints":
            # da_constraints: Download full CSV, then filter for new data only
            logging.info("Downloading full constraints CSV from NESO...")
            
            try:
                from pipeline.ingest.da_constraints import NESO_URL
                import requests
                from io import StringIO
                
                response = requests.get(NESO_URL, timeout=60)
                response.raise_for_status()
                
                all_data = pd.read_csv(StringIO(response.text))
                logging.info(f"Downloaded {len(all_data):,} total rows")
                
                # Convert dates
                all_data[date_col] = pd.to_datetime(all_data[date_col], errors='coerce')
                all_data = all_data.dropna(subset=[date_col])
                
                # Filter for new data only
                new_data = all_data[all_data[date_col] > last_date].copy()
                
                if new_data.empty:
                    logging.info(f"ℹ️ No new {dataset_name} data found (already up to date)")
                    return
                
                logging.info(f"Found {len(new_data):,} new rows after {last_date.date()}")
                
            except Exception as e:
                logging.error(f"Failed to download/parse constraints: {e}")
                raise
            
        elif dataset_name == "windfor":
            # windfor uses start_date/end_date string parameters
            new_data = fetch_wind_forecast(
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=latest_available_date.strftime('%Y-%m-%d')
            )
        else:
            logging.error(f"Unknown system-level dataset: {dataset_name}")
            return
        
        if new_data is None or new_data.empty:
            logging.info(f"ℹ️ No new {dataset_name} data returned from API")
            return

        # Save to intermediate file
        save_intermediate(new_data, dataset_name)
        logging.info(f"Fetched {len(new_data):,} new rows for {dataset_name}")
        
        # Merge to final file
        merge_intermediates_to_final(dataset_name, file_path)
        logging.info(f"✅ System-level incremental extraction complete for {dataset_name}")

    except Exception as e:
        logging.error(f"❌ Error in system-level extraction for {dataset_name}: {e}")
        logging.error(traceback.format_exc())
        raise

# ============================
# MAIN
# ============================
def run_incremental_extraction():
    """Main function to run incremental extraction for all datasets."""
    setup_logging()
    logging.info("🚀 Starting unified incremental extraction pipeline (Parquet)")
    logging.info(f"📅 API lag days configured: {API_LAG_DAYS}")
    
    start_time = datetime.now()
    datasets_processed = 0
    datasets_failed = 0

    for dataset_name, config in DATASETS.items():
        try:
            logging.info(f"\n{'='*50}")
            logging.info(f"Processing {dataset_name.upper()}")
            logging.info(f"{'='*50}")
            
            if config["is_bmu_level"]:
                extract_bmu_level_incremental(dataset_name, config["date_col"])
            else:
                extract_system_level_incremental(dataset_name, config["date_col"])
            
            datasets_processed += 1
            
        except Exception as e:
            datasets_failed += 1  # ✅ Fixed: Now properly increments
            logging.error(f"❌ Failed to process {dataset_name}: {e}")
            logging.error(traceback.format_exc())

    # Summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    logging.info(f"\n{'='*50}")
    logging.info(f"EXTRACTION SUMMARY")
    logging.info(f"{'='*50}")
    logging.info(f"✅ Datasets processed successfully: {datasets_processed}")
    logging.info(f"❌ Datasets failed: {datasets_failed}")
    logging.info(f"⏱️ Total duration: {duration}")
    logging.info(f"🎉 Incremental extraction pipeline completed!")


if __name__ == "__main__":
    run_incremental_extraction()