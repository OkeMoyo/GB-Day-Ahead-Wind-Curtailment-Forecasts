"""
Data cleaning script for national wind generation forecasts (WINDFOR).

Processes day-ahead wind forecast data from Elexon BMRS API v1.
- Uses total wind forecast (offshore + onshore sum)
- Keeps offshore/onshore components for future analysis
- Data is already at half-hourly resolution (48 settlement periods/day)
- Converts UTC timestamps to timezone-naive for consistency
- Handles DST transitions (46 periods in spring, 50 in fall)

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
    logging.info("="*60)
    logging.info("Starting WINDFOR data cleaning...")
    logging.info("="*60)

    # Load raw data
    try:
        windfor = pd.read_parquet(RAW_PATH)
        logging.info(f"✅ Loaded WINDFOR data: {windfor.shape[0]:,} rows × {windfor.shape[1]} columns")
        logging.info(f"   Date range: {windfor['settlementDate'].min()} to {windfor['settlementDate'].max()}")
    except Exception as e:
        logging.error(f"❌ Failed to load WINDFOR data from {RAW_PATH}: {e}")
        raise

    # Validate required columns exist
    required_cols = ['startTime', 'wind_offshore_forecast', 'wind_onshore_forecast', 'wind_total_forecast']
    missing_cols = set(required_cols) - set(windfor.columns)
    if missing_cols:
        logging.error(f"❌ Missing required columns: {missing_cols}")
        logging.error(f"   Available columns: {windfor.columns.tolist()}")
        raise ValueError(f"Missing columns: {missing_cols}")

    # Drop unnecessary columns
    windfor_cleaned = windfor.drop(
        columns=['publishTime', 'settlementDate', 'settlementPeriod'],
        errors='ignore'
    )
    logging.info(f"✅ Dropped unnecessary columns")

    # Rename columns for clarity and backward compatibility
    windfor_cleaned = windfor_cleaned.rename(columns={
        'startTime': 'half_hour_time',
        'wind_total_forecast': 'sys_wind_gen_forecast',
        'wind_offshore_forecast': 'wind_offshore_forecast',
        'wind_onshore_forecast': 'wind_onshore_forecast'
    })
    logging.info(f"✅ Renamed columns: {windfor_cleaned.columns.tolist()}")

    # Convert time column to timezone-naive datetime
    # API returns UTC (e.g., "2025-12-29T23:30:00Z")
    # Strip timezone for consistency with other data sources
    windfor_cleaned['half_hour_time'] = pd.to_datetime(
        windfor_cleaned['half_hour_time']
    ).dt.tz_localize(None)
    
    logging.info(f"✅ Converted 'half_hour_time' to timezone-naive datetime")
    logging.info(f"   Time range: {windfor_cleaned['half_hour_time'].min()} to {windfor_cleaned['half_hour_time'].max()}")

    # Data quality checks
    logging.info("")
    logging.info("Performing data quality checks...")
    
    # Check for missing values
    missing_counts = windfor_cleaned.isnull().sum()
    if missing_counts.any():
        logging.warning(f"⚠️  Missing values detected:")
        for col, count in missing_counts[missing_counts > 0].items():
            logging.warning(f"   - {col}: {count} missing ({count/len(windfor_cleaned)*100:.2f}%)")
    else:
        logging.info("✅ No missing values")

    # Check for duplicate timestamps
    duplicates = windfor_cleaned['half_hour_time'].duplicated().sum()
    if duplicates > 0:
        logging.warning(f"⚠️  {duplicates} duplicate timestamps detected - removing duplicates")
        windfor_cleaned = windfor_cleaned.drop_duplicates(subset=['half_hour_time'], keep='first')
    else:
        logging.info("✅ No duplicate timestamps")

    # Check for negative forecasts (should not happen)
    for col in ['sys_wind_gen_forecast', 'wind_offshore_forecast', 'wind_onshore_forecast']:
        negative_count = (windfor_cleaned[col] < 0).sum()
        if negative_count > 0:
            logging.warning(f"⚠️  {negative_count} negative values in '{col}' - setting to 0")
            windfor_cleaned.loc[windfor_cleaned[col] < 0, col] = 0
    
    logging.info("✅ No negative forecast values")

    # Sort by time
    windfor_cleaned = windfor_cleaned.sort_values('half_hour_time').reset_index(drop=True)
    logging.info("✅ Sorted data by time")

    # Reorder columns for clarity (time first, then total, then components)
    column_order = [
        'half_hour_time',
        'sys_wind_gen_forecast',
        'wind_offshore_forecast',
        'wind_onshore_forecast'
    ]
    windfor_cleaned = windfor_cleaned[column_order]

    # Summary statistics
    logging.info("")
    logging.info("Summary Statistics:")
    logging.info(f"   Total records: {len(windfor_cleaned):,}")
    logging.info(f"   Unique dates: {windfor_cleaned['half_hour_time'].dt.date.nunique():,}")
    logging.info(f"   Time range: {windfor_cleaned['half_hour_time'].min()} to {windfor_cleaned['half_hour_time'].max()}")
    logging.info(f"   Total wind forecast range: {windfor_cleaned['sys_wind_gen_forecast'].min():.1f} - {windfor_cleaned['sys_wind_gen_forecast'].max():.1f} MW")
    logging.info(f"   Offshore wind range: {windfor_cleaned['wind_offshore_forecast'].min():.1f} - {windfor_cleaned['wind_offshore_forecast'].max():.1f} MW")
    logging.info(f"   Onshore wind range: {windfor_cleaned['wind_onshore_forecast'].min():.1f} - {windfor_cleaned['wind_onshore_forecast'].max():.1f} MW")
    logging.info(f"   Mean total forecast: {windfor_cleaned['sys_wind_gen_forecast'].mean():.1f} MW")
    logging.info(f"   Median total forecast: {windfor_cleaned['sys_wind_gen_forecast'].median():.1f} MW")

    # Check time continuity (should be 48 records per day, except DST transitions)
    logging.info("")
    date_counts = windfor_cleaned['half_hour_time'].dt.date.value_counts().sort_index()
    
    # DST transition days can have 46 (spring forward) or 50 (fall back) periods
    acceptable_counts = {46, 47, 48, 49, 50}
    incomplete_days = date_counts[~date_counts.isin(acceptable_counts)]
    
    if len(incomplete_days) > 0:
        logging.warning(f"⚠️  {len(incomplete_days)} days with genuinely incomplete data:")
        for date, count in incomplete_days.head(10).items():
            logging.warning(f"   - {date}: {count} records (expected 46-50)")
        if len(incomplete_days) > 10:
            logging.warning(f"   ... and {len(incomplete_days) - 10} more")
    else:
        logging.info("✅ All days have acceptable record counts")

    # Identify DST transition days for informational purposes
    dst_days = date_counts[(date_counts == 46) | (date_counts == 50)]
    if len(dst_days) > 0:
        logging.info(f"ℹ️  Detected {len(dst_days)} DST transition days (expected):")
        for date, count in dst_days.items():
            transition_type = "Spring forward (clocks ahead)" if count == 46 else "Fall back (clocks behind)"
            logging.info(f"   - {date}: {count} periods ({transition_type})")

    # Save cleaned data
    logging.info("")
    try:
        windfor_cleaned.to_parquet(OUTPUT_PATH, index=False)
        file_size_mb = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)
        logging.info(f"💾 Saved cleaned WINDFOR data to {OUTPUT_PATH}")
        logging.info(f"   File size: {file_size_mb:.2f} MB")
    except Exception as e:
        logging.error(f"❌ Failed to save cleaned WINDFOR data: {e}")
        raise

    logging.info("")
    logging.info("="*60)
    logging.info("✅ WINDFOR data cleaning completed successfully")
    logging.info("="*60)
    
    return windfor_cleaned

if __name__ == "__main__":
    clean_windfor()