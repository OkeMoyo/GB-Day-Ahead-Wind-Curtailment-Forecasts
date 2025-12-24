"""
Build inference-ready forecast features WITHOUT FPN/BOALF dependency.

This script creates a separate dataset optimized for operational forecasting:
- Uses only forecast data (wind, demand, constraints)
- No historical curtailment data required
- Generates system-level features for all available forecast dates
- Output: data/processed/forecast_features.parquet

Usage:
    python -m pipeline.preprocessing.build_forecast_features
"""

import logging
from pathlib import Path
from datetime import timedelta

import numpy as np
import pandas as pd

# -----------------------
# Logging
# -----------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("build_forecast_features")

# -----------------------
# Paths
# -----------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

WINDFOR_PATH = DATA_PROCESSED_DIR / "windfor" / "windfor_cleaned.parquet"
DEMANDFOR_PATH = DATA_PROCESSED_DIR / "demandfor" / "demandfor_cleaned.parquet"
CONSTRAINTS_PATH = DATA_PROCESSED_DIR / "da_constraints" / "da_constraints_cleaned.parquet"
BMUS_PATH = DATA_PROCESSED_DIR / "bmus" / "bmus_cleaned.parquet"

OUTPUT_PATH = DATA_PROCESSED_DIR / "forecast_features.parquet"

# -----------------------
# Helper Functions
# -----------------------
def add_cyclical_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cyclical time features."""
    df['hour'] = pd.to_datetime(df['settlement_period_time']).dt.hour
    df['month'] = pd.to_datetime(df['settlement_period_time']).dt.month
    
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    df = df.drop(columns=['hour', 'month'], errors='ignore')
    return df

def get_common_dates(windfor: pd.DataFrame, demandfor: pd.DataFrame, constraints: pd.DataFrame) -> list:
    """Find dates with complete data across all forecast sources."""
    log.info("Finding common dates with complete forecast data...")
    
    # Wind forecast dates (48 periods per day)
    windfor['date'] = pd.to_datetime(windfor['half_hour_time']).dt.date
    wind_dates = windfor.groupby('date')['half_hour_time'].nunique()
    complete_wind_dates = set(wind_dates[wind_dates == 48].index)
    log.info(f"   Wind forecast: {len(complete_wind_dates)} complete days")
    
    # Demand forecast dates (48 periods per day)
    demandfor['date'] = pd.to_datetime(demandfor['half_hour_time']).dt.date
    demand_dates = demandfor.groupby('date')['half_hour_time'].nunique()
    complete_demand_dates = set(demand_dates[demand_dates == 48].index)
    log.info(f"   Demand forecast: {len(complete_demand_dates)} complete days")
    
    # Constraint dates (daily resolution)
    constraints['date'] = pd.to_datetime(constraints['Date (GMT/BST)']).dt.date
    constraint_dates = set(constraints['date'].unique())
    log.info(f"   Constraints: {len(constraint_dates)} days")
    
    # Find intersection
    common_dates = complete_wind_dates & complete_demand_dates & constraint_dates
    common_dates = sorted(list(common_dates))
    
    log.info(f"   Common dates across all sources: {len(common_dates)} days")
    
    if common_dates:
        log.info(f"   Date range: {common_dates[0]} to {common_dates[-1]}")
    
    return common_dates

def build_features_for_date(date, windfor, demandfor, constraints, bmus_capacity, n_bmus):
    """Build system-level features for a single date."""
    prediction_start = pd.Timestamp(date)
    prediction_end = prediction_start + timedelta(days=1)
    
    # Generate 48 timestamps
    timestamps = pd.date_range(
        start=prediction_start,
        end=prediction_end,
        freq='30min',
        inclusive='left'
    )
    
    df = pd.DataFrame({'settlement_period_time': timestamps})
    
    # Merge wind forecast
    windfor_day = windfor[
        (windfor['half_hour_time'] >= prediction_start) &
        (windfor['half_hour_time'] < prediction_end)
    ].copy()
    
    df = df.merge(
        windfor_day[['half_hour_time', 'sys_wind_gen_forecast']],
        left_on='settlement_period_time',
        right_on='half_hour_time',
        how='left'
    ).drop(columns=['half_hour_time'])
    
    # Merge demand forecast
    demandfor_day = demandfor[
        (demandfor['half_hour_time'] >= prediction_start) &
        (demandfor['half_hour_time'] < prediction_end)
    ].copy()
    
    df = df.merge(
        demandfor_day[['half_hour_time', 'sys_demand_forecast']],
        left_on='settlement_period_time',
        right_on='half_hour_time',
        how='left'
    ).drop(columns=['half_hour_time'])
    
    # Merge constraints (daily)
    constraints_day = constraints[constraints['date'] == date]
    
    if len(constraints_day) > 0:
        for col in constraints_day.columns:
            if col not in ['Date (GMT/BST)', 'date']:
                df[col] = constraints_day[col].iloc[0]
    
    # Add BMU metadata
    df['generationCapacity'] = bmus_capacity
    df['bmUnit'] = n_bmus
    
    return df

# -----------------------
# Main Function
# -----------------------
def main():
    """Build forecast features for all available dates."""
    log.info("=" * 60)
    log.info("BUILDING FORECAST FEATURES (INFERENCE-READY)")
    log.info("=" * 60)
    
    # Load data
    log.info("\nLoading forecast data sources...")
    windfor = pd.read_parquet(WINDFOR_PATH)
    demandfor = pd.read_parquet(DEMANDFOR_PATH)
    constraints = pd.read_parquet(CONSTRAINTS_PATH)
    bmus = pd.read_parquet(BMUS_PATH)
    
    windfor['half_hour_time'] = pd.to_datetime(windfor['half_hour_time'])
    demandfor['half_hour_time'] = pd.to_datetime(demandfor['half_hour_time'])
    constraints['Date (GMT/BST)'] = pd.to_datetime(constraints['Date (GMT/BST)'])
    
    bmus_capacity = bmus['generationCapacity'].sum()
    n_bmus = len(bmus)
    
    log.info(f"   Wind forecast: {len(windfor):,} rows")
    log.info(f"   Demand forecast: {len(demandfor):,} rows")
    log.info(f"   Constraints: {len(constraints):,} rows")
    log.info(f"   System capacity: {bmus_capacity:,.0f} MW across {n_bmus} BMUs")
    
    # Find common dates
    common_dates = get_common_dates(windfor, demandfor, constraints)
    
    if not common_dates:
        log.error("[ERROR] No common dates found across forecast sources!")
        return
    
    # Build features for each date
    log.info(f"\nBuilding features for {len(common_dates)} dates...")
    
    all_features = []
    
    for i, date in enumerate(common_dates, 1):
        if i % 50 == 0 or i == len(common_dates):
            log.info(f"   Processing date {i}/{len(common_dates)}: {date}")
        
        try:
            df_date = build_features_for_date(
                date, windfor, demandfor, constraints, 
                bmus_capacity, n_bmus
            )
            
            # Validate 48 periods
            if len(df_date) != 48:
                log.warning(f"   ⚠️ Skipping {date}: only {len(df_date)}/48 periods")
                continue
            
            all_features.append(df_date)
            
        except Exception as e:
            log.warning(f"   ⚠️ Error processing {date}: {e}")
            continue
    
    if not all_features:
        log.error("[ERROR] No features generated!")
        return
    
    # Combine all dates
    log.info("\nCombining features from all dates...")
    forecast_features = pd.concat(all_features, ignore_index=True)
    
    # Add temporal features
    log.info("Adding temporal features...")
    forecast_features = add_cyclical_time_features(forecast_features)
    
    # Summary
    log.info("\n" + "=" * 60)
    log.info("FEATURE SUMMARY")
    log.info("=" * 60)
    log.info(f"Total rows: {len(forecast_features):,}")
    log.info(f"Total dates: {forecast_features['settlement_period_time'].dt.date.nunique()}")
    log.info(f"Date range: {forecast_features['settlement_period_time'].min().date()} to {forecast_features['settlement_period_time'].max().date()}")
    log.info(f"Columns: {len(forecast_features.columns)}")
    
    missing = forecast_features.isnull().sum().sum()
    log.info(f"Missing values: {missing:,} ({100*missing/forecast_features.size:.2f}%)")
    
    # Save
    log.info(f"\nSaving to {OUTPUT_PATH}...")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    forecast_features.to_parquet(OUTPUT_PATH, index=False)
    
    log.info("=" * 60)
    log.info("✅ FORECAST FEATURES BUILD COMPLETED SUCCESSFULLY")
    log.info("=" * 60)
    log.info(f"Output: {OUTPUT_PATH}")
    log.info(f"Size: {OUTPUT_PATH.stat().st_size / 1024 / 1024:.2f} MB")

if __name__ == "__main__":
    main()