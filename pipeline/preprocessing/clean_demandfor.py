# clean_demandfor.py
"""
Data cleaning script for national demand forecasts (DEMANDFOR).
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
RAW_PATH = os.path.join("data", "raw", "demandfor", "demandfor.parquet")
PROCESSED_DIR = os.path.join("data", "processed", "demandfor")
os.makedirs(PROCESSED_DIR, exist_ok=True)
OUTPUT_PATH = os.path.join(PROCESSED_DIR, "demandfor_cleaned.parquet")

# -------------------------------
# Helper functions
# -------------------------------
def parse_cp_time(cp_time):
    """Convert CP time string like '30', '230' to minutes past midnight."""
    cp_time_str = str(int(cp_time)).zfill(4)
    hour = int(cp_time_str[:-2])
    minute = int(cp_time_str[-2:])
    return hour * 60 + minute

def clean_demandfor():
    """Load, clean, and save national demand forecast dataset."""
    logging.info("Starting DEMANDFOR data cleaning...")

    # Load raw data
    try:
        demandfor = pd.read_parquet(RAW_PATH)
        logging.info(f"Loaded DEMANDFOR data with shape {demandfor.shape}")
    except Exception as e:
        logging.error(f"Failed to load DEMANDFOR data from {RAW_PATH}: {e}")
        raise

    # Ensure TARGETDATE is datetime
    demandfor['TARGETDATE'] = pd.to_datetime(demandfor['TARGETDATE'])

    all_days_rows = []

    # Process each day individually
    for day, group in demandfor.groupby('TARGETDATE'):
        # Generate all 48 half-hour slots for the day
        half_hour_times = [pd.Timestamp(day) + pd.Timedelta(minutes=30 * i) for i in range(48)] # type: ignore
        slot_dict = {h: None for h in half_hour_times}

        group = group.sort_values(['CP_ST_TIME', 'CP_END_TIME']).reset_index(drop=True)
        n = len(group)

        # Assign forecast values to half-hour intervals
        for i, row in group.iterrows():
            st_min = parse_cp_time(row['CP_ST_TIME'])
            end_min = parse_cp_time(row['CP_END_TIME'])

            if st_min == end_min:
                next_st_min = None
                if i + 1 < n: # type: ignore
                    next_st_min = parse_cp_time(group.iloc[i + 1]['CP_ST_TIME']) # type: ignore
                interval_end = next_st_min if next_st_min is not None else st_min + 30
            else:
                interval_end = end_min

            half_hours = list(range(st_min, interval_end, 30))
            for offset_min in half_hours:
                half_hour_time = pd.Timestamp(day) + pd.Timedelta(minutes=offset_min) # type: ignore
                slot_dict[half_hour_time] = row['FORECASTDEMAND']

        # Forward fill missing slots (fill first slot with first available forecast)
        slot_times = sorted(slot_dict.keys())
        values = [slot_dict[t] for t in slot_times]
        first_val = next((v for v in values if v is not None), None)

        filled_values = []
        prev_val = first_val
        for v in values:
            if v is not None:
                prev_val = v
            filled_values.append(prev_val)

        # Build rows
        for t, v in zip(slot_times, filled_values):
            all_days_rows.append({
                'TARGETDATE': day,
                'half_hour_time': t,
                'sys_demand_forecast': v
            })

    # Create cleaned DataFrame
    demandfor_cleaned = pd.DataFrame(all_days_rows)
    demandfor_cleaned = demandfor_cleaned.sort_values(
        ['TARGETDATE', 'half_hour_time']
    ).reset_index(drop=True)

    logging.info(f"Transformed demand forecast to half-hourly resolution. New shape: {demandfor_cleaned.shape}")

    # Save cleaned data
    try:
        demandfor_cleaned.to_parquet(OUTPUT_PATH, index=False)
        logging.info(f"Saved cleaned DEMANDFOR data to {OUTPUT_PATH}")
    except Exception as e:
        logging.error(f"Failed to save cleaned DEMANDFOR data: {e}")
        raise

    logging.info("DEMANDFOR data cleaning completed successfully.")
    return demandfor_cleaned

if __name__ == "__main__":
    clean_demandfor()