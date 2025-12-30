#!/usr/bin/env python3
"""
windfor.py

Fetch day-ahead wind generation forecasts from Elexon BMRS API.
- Extracts Wind Offshore and Wind Onshore separately
- Aggregates into wind_total_forecast for model compatibility
- Filters out solar generation data
- Handles API's 7-day maximum range limit by batching requests
- Raw extraction only (cleaning done in clean_windfor.py)

Run:
    python -m pipeline.ingest.windfor
"""

import os
import time
import logging
import requests
import pandas as pd
from datetime import datetime, timedelta

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
DATA_DIR = "data/raw/windfor"
OUTPUT_FILE = os.path.join(DATA_DIR, "windfor.parquet")

# Date range for initial extraction
START_DATE = "2024-01-01"
END_DATE = datetime.now().strftime("%Y-%m-%d")

# New BMRS API endpoint (v1)
API_URL = "https://data.elexon.co.uk/bmrs/api/v1/forecast/generation/wind-and-solar/day-ahead"

# API constraints
MAX_DATE_RANGE_DAYS = 7  # API maximum is 7 days per request

# Retry configuration
MAX_RETRIES = 3
BACKOFF_FACTOR = 2  # Exponential backoff
REQUEST_DELAY = 0.5  # Delay between requests to avoid rate limiting

# -------------------------------------------------------------------
# Logging Setup
# -------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

# -------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------
def generate_date_ranges(start_date: str, end_date: str, chunk_days: int = 7):
    """
    Generate date range chunks for API requests.
    
    ✅ FIXED: Requests chunk_days + 1 to handle API's incomplete end date behavior.
    API only returns settlement period 1 for the 'to' date, so we request an extra
    day and accept 1-day overlap between chunks. Duplicates removed during processing.
    
    Args:
        start_date: Start date string 'YYYY-MM-DD'
        end_date: End date string 'YYYY-MM-DD'
        chunk_days: Desired days per chunk (default 7)
    
    Yields:
        tuple: (chunk_start_date, chunk_end_date) as strings
    """
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    
    current = start
    while current <= end:
        # Request chunk_days + 1 to ensure we get full data for chunk_days
        # The 'to' date only returns partial data (settlement period 1)
        chunk_end = min(current + timedelta(days=chunk_days), end)  # +1 day
        yield (
            current.strftime("%Y-%m-%d"),
            chunk_end.strftime("%Y-%m-%d")
        )
        # Advance by chunk_days (not chunk_days + 1) to create 1-day overlap
        current = current + timedelta(days=chunk_days)


# -------------------------------------------------------------------
# Core Functions
# -------------------------------------------------------------------
def fetch_wind_forecast_chunk(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch day-ahead wind generation forecast for a single date range chunk.
    
    Args:
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
    
    Returns:
        pd.DataFrame: Raw wind forecast data with offshore/onshore split
    """
    params = {
        "from": start_date,
        "to": end_date,
        "processType": "day ahead"  # Filter for day-ahead only
    }
    
    logging.info(f"  Fetching chunk: {start_date} to {end_date}")
    
    retries = 0
    while retries < MAX_RETRIES:
        try:
            response = requests.get(
                API_URL, 
                params=params, 
                headers={"accept": "application/json"},
                timeout=30
            )
            response.raise_for_status()
            
            json_data = response.json()
            
            # Extract data array from response
            if "data" not in json_data:
                logging.error("    API response missing 'data' key")
                return pd.DataFrame()
            
            records = json_data["data"]
            
            if not records:
                logging.warning(f"    No data returned for {start_date} to {end_date}")
                return pd.DataFrame()
            
            # Filter for wind generation only (exclude solar)
            wind_records = [
                record for record in records 
                if record.get("businessType") == "Wind generation"
            ]
            
            if not wind_records:
                logging.warning("    No wind generation data found (only solar returned)")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(wind_records)
            
            # ✅ DIAGNOSTIC: Check date coverage
            unique_dates = sorted(df['settlementDate'].unique())
            logging.info(f"    Retrieved {len(wind_records)} wind records")
            logging.info(f"    Date coverage: {unique_dates[0]} to {unique_dates[-1]} ({len(unique_dates)} unique dates)")
            
            # Check for problematic boundary dates
            date_period_counts = df.groupby('settlementDate')['settlementPeriod'].nunique()
            partial_dates = date_period_counts[date_period_counts < 48]
            if len(partial_dates) > 0:
                logging.warning(f"    ⚠️  Partial data for dates: {list(partial_dates.index)}")
            
            return df
            
        except requests.exceptions.HTTPError as e:
            logging.error(f"    HTTP error: {e}")
            if response.status_code == 400:
                logging.error(f"    Bad request for range {start_date} to {end_date}")
                logging.error(f"    Response: {response.text[:200]}")
                return pd.DataFrame()
            retries += 1
            
        except requests.exceptions.RequestException as e:
            retries += 1
            wait_time = BACKOFF_FACTOR ** (retries - 1)
            logging.warning(
                f"    Request failed (retry {retries}/{MAX_RETRIES}): {e}"
            )
            if retries < MAX_RETRIES:
                logging.info(f"    Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
        
        except Exception as e:
            logging.error(f"    Unexpected error: {e}")
            return pd.DataFrame()
    
    logging.error(f"    Failed to fetch chunk {start_date} to {end_date} after maximum retries")
    return pd.DataFrame()


def fetch_wind_forecast(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch day-ahead wind generation forecast across full date range.
    
    Handles API's 7-day limit by splitting into multiple requests.
    
    Args:
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
    
    Returns:
        pd.DataFrame: Combined wind forecast data from all chunks
    """
    logging.info(f"Fetching wind forecast from {start_date} to {end_date}")
    logging.info(f"API URL: {API_URL}")
    
    # Calculate date range chunks
    date_ranges = list(generate_date_ranges(start_date, end_date, MAX_DATE_RANGE_DAYS))
    total_chunks = len(date_ranges)
    
    logging.info(f"Date range split into {total_chunks} chunks (7-day windows with 1-day overlap)")
    logging.info("")
    
    all_data = []
    
    for i, (chunk_start, chunk_end) in enumerate(date_ranges, 1):
        logging.info(f"[{i}/{total_chunks}] Processing chunk {chunk_start} to {chunk_end}")
        
        chunk_df = fetch_wind_forecast_chunk(chunk_start, chunk_end)
        
        if not chunk_df.empty:
            all_data.append(chunk_df)
            logging.info(f"    ✅ Chunk {i} successful ({len(chunk_df)} records)")
        else:
            logging.warning(f"    ⚠️  Chunk {i} returned no data")
        
        # Rate limiting delay (except for last request)
        if i < total_chunks:
            time.sleep(REQUEST_DELAY)
        
        logging.info("")
    
    if not all_data:
        logging.error("No data retrieved from any chunk")
        return pd.DataFrame()
    
    # Combine all chunks
    df_combined = pd.concat(all_data, ignore_index=True)
    
    logging.info(f"Combined data from {len(all_data)} successful chunks")
    logging.info(f"Total records before deduplication: {len(df_combined):,}")
    logging.info(f"  - Offshore records: {sum(1 for _, r in df_combined.iterrows() if r.get('psrType') == 'Wind Offshore')}")
    logging.info(f"  - Onshore records: {sum(1 for _, r in df_combined.iterrows() if r.get('psrType') == 'Wind Onshore')}")
    logging.info("")
    
    # Process and aggregate wind data
    df_processed = process_wind_data(df_combined)
    
    return df_processed


def process_wind_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw wind data into aggregated format.
    
    Creates separate columns for offshore/onshore and total wind.
    Groups by settlement date/period to align offshore and onshore on same row.
    
    Args:
        df: Raw DataFrame with psrType (Wind Offshore/Onshore) and quantity
    
    Returns:
        pd.DataFrame: Processed data with columns:
            - settlementDate
            - settlementPeriod
            - startTime
            - publishTime
            - wind_offshore_forecast (MW)
            - wind_onshore_forecast (MW)
            - wind_total_forecast (MW) - sum of offshore + onshore
    """
    logging.info("Processing and aggregating wind data...")
    
    # ✅ Remove exact duplicates that may occur from overlapping chunks
    initial_count = len(df)
    df = df.drop_duplicates(subset=['settlementDate', 'settlementPeriod', 'psrType', 'quantity'])
    duplicates_removed = initial_count - len(df)
    if duplicates_removed > 0:
        logging.info(f"   Removed {duplicates_removed} duplicate records from overlapping chunks")
    
    # Pivot to get offshore and onshore as separate columns
    df_pivot = df.pivot_table(
        index=["settlementDate", "settlementPeriod", "startTime", "publishTime"],
        columns="psrType",
        values="quantity",
        aggfunc="first"  # Take first value if duplicates exist
    ).reset_index()
    
    # Rename columns to match expected schema
    df_pivot.columns.name = None  # Remove 'psrType' as column level name
    
    # Create standardized column names
    column_mapping = {
        "Wind Offshore": "wind_offshore_forecast",
        "Wind Onshore": "wind_onshore_forecast"
    }
    df_pivot.rename(columns=column_mapping, inplace=True)
    
    # Ensure both columns exist (fill with 0 if missing)
    if "wind_offshore_forecast" not in df_pivot.columns:
        df_pivot["wind_offshore_forecast"] = 0.0
    if "wind_onshore_forecast" not in df_pivot.columns:
        df_pivot["wind_onshore_forecast"] = 0.0
    
    # Calculate total wind forecast (sum of offshore + onshore)
    df_pivot["wind_total_forecast"] = (
        df_pivot["wind_offshore_forecast"] + 
        df_pivot["wind_onshore_forecast"]
    )
    
    # Select and order columns
    output_columns = [
        "publishTime",
        "settlementDate",
        "settlementPeriod",
        "startTime",
        "wind_offshore_forecast",
        "wind_onshore_forecast",
        "wind_total_forecast"
    ]
    
    df_final = df_pivot[output_columns].copy()
    
    # ✅ Remove any remaining duplicate settlement periods (keep first)
    df_final = df_final.drop_duplicates(subset=['settlementDate', 'settlementPeriod'], keep='first')
    
    # Sort by settlement date and period
    df_final = df_final.sort_values(["settlementDate", "settlementPeriod"]).reset_index(drop=True)
    
    # ✅ DIAGNOSTIC: Check for missing settlement periods
    date_period_counts = df_final.groupby('settlementDate').size()
    incomplete_dates = date_period_counts[date_period_counts != 48]
    if len(incomplete_dates) > 0:
        logging.warning(f"   ⚠️  {len(incomplete_dates)} dates with incomplete data after processing:")
        for date, count in list(incomplete_dates.items())[:5]:
            logging.warning(f"      {date}: {count} periods (expected 48)")
        if len(incomplete_dates) > 5:
            logging.warning(f"      ... and {len(incomplete_dates) - 5} more")
    else:
        logging.info("   ✅ All dates have complete 48 settlement periods")
    
    logging.info(f"✅ Processed {len(df_final):,} settlement periods")
    logging.info(f"   Date range: {df_final['settlementDate'].min()} to {df_final['settlementDate'].max()}")
    logging.info(f"   Offshore range: {df_final['wind_offshore_forecast'].min():.1f} - {df_final['wind_offshore_forecast'].max():.1f} MW")
    logging.info(f"   Onshore range: {df_final['wind_onshore_forecast'].min():.1f} - {df_final['wind_onshore_forecast'].max():.1f} MW")
    logging.info(f"   Total range: {df_final['wind_total_forecast'].min():.1f} - {df_final['wind_total_forecast'].max():.1f} MW")
    
    return df_final


def save_wind_forecast(df: pd.DataFrame, output_path: str):
    """
    Save wind forecast DataFrame to Parquet file.
    
    Overwrites existing file to ensure clean data.
    
    Args:
        df: Processed wind forecast data
        output_path: Path to output Parquet file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    df.to_parquet(output_path, index=False, engine="pyarrow")
    
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    logging.info(f"💾 Saved {len(df):,} records to {output_path}")
    logging.info(f"   File size: {file_size_mb:.2f} MB")


def run_extraction():
    """
    Main extraction pipeline for wind forecast data.
    
    Fetches data from BMRS API and saves to Parquet.
    Overwrites existing windfor.parquet file.
    """
    logging.info("="*60)
    logging.info("WIND FORECAST EXTRACTION - FULL REFRESH")
    logging.info("="*60)
    logging.info(f"Date range: {START_DATE} to {END_DATE}")
    logging.info("")
    
    # Fetch data (with automatic chunking)
    df = fetch_wind_forecast(START_DATE, END_DATE)
    
    if df.empty:
        logging.error("❌ No wind forecast data retrieved - extraction failed")
        return
    
    # Validate data
    required_columns = [
        "settlementDate", "settlementPeriod", 
        "wind_offshore_forecast", "wind_onshore_forecast", "wind_total_forecast"
    ]
    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        logging.error(f"❌ Missing required columns: {missing_cols}")
        return
    
    # Save to file
    save_wind_forecast(df, OUTPUT_FILE)
    
    logging.info("")
    logging.info("="*60)
    logging.info("✅ WIND FORECAST EXTRACTION COMPLETED")
    logging.info("="*60)


# -------------------------------------------------------------------
# Entry Point
# -------------------------------------------------------------------
if __name__ == "__main__":
    run_extraction()