"""
Day-ahead curtailment forecasting inference pipeline.

Generates predictions for all 48 settlement periods using:
- Latest trained XGBoost model
- Latest available forecast data (wind, demand, constraints) ONLY
- NO dependency on historical FPN/BOALF data

Outputs:
- predictions/YYYYMMDD_predictions.csv (minimal format)
- predictions/YYYYMMDD_metadata.json (run metadata)

Usage:
    python -m pipeline.inference
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
import joblib

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Create logs directory if it doesn't exist
LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------
# Logging (Windows-compatible encoding)
# -----------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOGS_DIR / "inference.log", mode="a", encoding='utf-8')
    ]
)
log = logging.getLogger("inference")

# -----------------------
# Paths
# -----------------------
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
PREDICTIONS_DIR = PROJECT_ROOT / "predictions"
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

# Forecast data sources (NOT features.parquet!)
WINDFOR_PATH = DATA_PROCESSED_DIR / "windfor" / "windfor_cleaned.parquet"
DEMANDFOR_PATH = DATA_PROCESSED_DIR / "demandfor" / "demandfor_cleaned.parquet"
CONSTRAINTS_PATH = DATA_PROCESSED_DIR / "da_constraints" / "da_constraints_cleaned.parquet"
BMUS_PATH = DATA_PROCESSED_DIR / "bmus" / "bmus_cleaned.parquet"

# Model artifacts
MODEL_FILE = MODELS_DIR / "sys-day-ahead-xgb-optimal.pkl"
FEATURES_FILE = MODELS_DIR / "sys-day-ahead-selected-features.pkl"
THRESHOLD_FILE = MODELS_DIR / "sys-day-ahead-threshold.json"

# -----------------------
# 1. Load Model Artifacts
# -----------------------
def load_model_artifacts() -> Tuple[xgb.XGBClassifier, List[str], float]:
    """Load trained model, selected features, and probability threshold."""
    log.info("=" * 60)
    log.info("LOADING MODEL ARTIFACTS")
    log.info("=" * 60)
    
    # Check if all files exist
    missing_files = []
    for file_path in [MODEL_FILE, FEATURES_FILE, THRESHOLD_FILE]:
        if not file_path.exists():
            missing_files.append(str(file_path))
    
    if missing_files:
        log.error(f"[ERROR] Missing model artifacts: {missing_files}")
        raise FileNotFoundError(f"Required model files not found: {missing_files}")
    
    # Load model
    log.info(f"Loading model from {MODEL_FILE}")
    model = joblib.load(MODEL_FILE)
    log.info(f"[OK] Model loaded (XGBoost version: {xgb.__version__})")
    
    # Load features
    log.info(f"Loading feature list from {FEATURES_FILE}")
    features = joblib.load(FEATURES_FILE)
    log.info(f"[OK] Loaded {len(features)} features")
    
    # Load threshold
    log.info(f"Loading threshold from {THRESHOLD_FILE}")
    with open(THRESHOLD_FILE, 'r') as f:
        threshold_data = json.load(f)
    threshold = threshold_data['threshold']
    log.info(f"[OK] Loaded threshold: {threshold:.4f} (precision: {threshold_data['precision']:.4f}, recall: {threshold_data['recall']:.4f})")
    
    return model, features, threshold

# -----------------------
# 2. Determine Prediction Date
# -----------------------
def get_prediction_date() -> datetime.date:
    """
    Find the latest date where ALL forecast features have complete data (48 settlement periods).
    
    Checks wind, demand, and constraint forecasts independently.
    Returns the most recent date where all forecasts are complete.
    """
    log.info("=" * 60)
    log.info("DETERMINING PREDICTION DATE FROM FORECAST DATA")
    log.info("=" * 60)
    
    dates = []
    
    # Check wind forecast
    if WINDFOR_PATH.exists():
        log.info(f"Checking wind forecast: {WINDFOR_PATH}")
        windfor = pd.read_parquet(WINDFOR_PATH, columns=['half_hour_time'])
        windfor['date'] = pd.to_datetime(windfor['half_hour_time']).dt.date
        wind_dates = windfor.groupby('date')['half_hour_time'].nunique()
        complete_wind_dates = wind_dates[wind_dates == 48].index.tolist()
        
        if complete_wind_dates:
            latest_wind = max(complete_wind_dates)
            dates.append(latest_wind)
            log.info(f"   Latest complete wind forecast: {latest_wind}")
        else:
            log.warning("   No complete wind forecast dates found!")
    else:
        log.error(f"   Wind forecast file not found: {WINDFOR_PATH}")
    
    # Check demand forecast
    if DEMANDFOR_PATH.exists():
        log.info(f"Checking demand forecast: {DEMANDFOR_PATH}")
        demandfor = pd.read_parquet(DEMANDFOR_PATH, columns=['half_hour_time'])
        demandfor['date'] = pd.to_datetime(demandfor['half_hour_time']).dt.date
        demand_dates = demandfor.groupby('date')['half_hour_time'].nunique()
        complete_demand_dates = demand_dates[demand_dates == 48].index.tolist()
        
        if complete_demand_dates:
            latest_demand = max(complete_demand_dates)
            dates.append(latest_demand)
            log.info(f"   Latest complete demand forecast: {latest_demand}")
        else:
            log.warning("   No complete demand forecast dates found!")
    else:
        log.error(f"   Demand forecast file not found: {DEMANDFOR_PATH}")
    
    # Check constraints (daily resolution - just need the date to exist)
    if CONSTRAINTS_PATH.exists():
        log.info(f"Checking constraint forecasts: {CONSTRAINTS_PATH}")
        constraints = pd.read_parquet(CONSTRAINTS_PATH, columns=['Date (GMT/BST)'])
        constraints['date'] = pd.to_datetime(constraints['Date (GMT/BST)']).dt.date
        constraint_dates = sorted(constraints['date'].unique())
        
        if constraint_dates:
            latest_constraint = constraint_dates[-1]
            dates.append(latest_constraint)
            log.info(f"   Latest constraint forecast: {latest_constraint}")
        else:
            log.warning("   No constraint forecast dates found!")
    else:
        log.error(f"   Constraint file not found: {CONSTRAINTS_PATH}")
    
    # Find minimum date across all forecasts (conservative approach)
    dates = [d for d in dates if d is not None]
    
    if not dates:
        log.error("[ERROR] No forecast data found in any source!")
        raise ValueError("Cannot determine prediction date: no forecast data available")
    
    # Use the earliest complete date (ensures all forecasts are available)
    prediction_date = min(dates)
    
    log.info(f"\n[OK] Selected prediction date: {prediction_date}")
    log.info(f"   (Latest date with complete forecasts across all sources)")
    
    if len(set(dates)) > 1:
        log.warning(f"WARNING: Forecast dates differ across sources: {sorted(set(dates))}")
        log.warning(f"   Using conservative date: {prediction_date}")
    
    return prediction_date

# -----------------------
# 3. Build Inference Features
# -----------------------
def add_cyclical_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cyclical time features (same as training)."""
    if 'hour' not in df.columns or 'month' not in df.columns:
        df['hour'] = pd.to_datetime(df['settlement_period_time']).dt.hour
        df['month'] = pd.to_datetime(df['settlement_period_time']).dt.month
    
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    df = df.drop(columns=['hour', 'month'], errors='ignore')
    return df

def build_inference_features(prediction_date: datetime.date, required_features: List[str]) -> pd.DataFrame:
    """
    Build system-level feature set for prediction date using ONLY forecast data.
    
    Does NOT require FPN/BOALF (historical curtailment data).
    Constructs features from scratch for operational forecasting.
    
    Features:
    - Removes duplicate timestamps (keeps first occurrence)
    - Backfills missing values from previous day's data
    """
    log.info("=" * 60)
    log.info("BUILDING INFERENCE FEATURES FROM FORECASTS")
    log.info("=" * 60)
    
    # Load forecast data sources
    log.info("Loading forecast data sources...")
    windfor = pd.read_parquet(WINDFOR_PATH)
    demandfor = pd.read_parquet(DEMANDFOR_PATH)
    constraints = pd.read_parquet(CONSTRAINTS_PATH)
    bmus = pd.read_parquet(BMUS_PATH)
    
    log.info(f"   Wind forecast: {len(windfor):,} rows")
    log.info(f"   Demand forecast: {len(demandfor):,} rows")
    log.info(f"   Constraints: {len(constraints):,} rows")
    log.info(f"   BMUs: {len(bmus):,} units")
    
    # Generate 48 half-hourly timestamps for the prediction date
    prediction_start = pd.Timestamp(prediction_date)
    prediction_end = prediction_start + timedelta(days=1)
    
    timestamps = pd.date_range(
        start=prediction_start,
        end=prediction_end,
        freq='30min',
        inclusive='left'  # Excludes the end timestamp (00:00 next day)
    )
    
    # Create base dataframe with all 48 periods
    df = pd.DataFrame({
        'settlement_period_time': timestamps
    })
    
    log.info(f"\nCreated base dataframe with {len(df)} settlement periods for {prediction_date}")
    
    # --- MERGE WIND FORECAST ---
    log.info("Merging wind forecast...")
    windfor['half_hour_time'] = pd.to_datetime(windfor['half_hour_time'])
    windfor_filtered = windfor[
        (windfor['half_hour_time'] >= prediction_start) &
        (windfor['half_hour_time'] < prediction_end)
    ].copy()
    
    log.info(f"   Found {len(windfor_filtered)} wind forecast records for {prediction_date}")
    
    df = df.merge(
        windfor_filtered[['half_hour_time', 'sys_wind_gen_forecast']],
        left_on='settlement_period_time',
        right_on='half_hour_time',
        how='left'
    ).drop(columns=['half_hour_time'])
    
    # --- MERGE DEMAND FORECAST ---
    log.info("Merging demand forecast...")
    demandfor['half_hour_time'] = pd.to_datetime(demandfor['half_hour_time'])
    demandfor_filtered = demandfor[
        (demandfor['half_hour_time'] >= prediction_start) &
        (demandfor['half_hour_time'] < prediction_end)
    ].copy()
    
    log.info(f"   Found {len(demandfor_filtered)} demand forecast records for {prediction_date}")
    
    df = df.merge(
        demandfor_filtered[['half_hour_time', 'sys_demand_forecast']],
        left_on='settlement_period_time',
        right_on='half_hour_time',
        how='left'
    ).drop(columns=['half_hour_time'])
    
    # --- REMOVE DUPLICATE TIMESTAMPS ---
    log.info("\nChecking for duplicate timestamps...")
    initial_len = len(df)
    df = df.drop_duplicates(subset=['settlement_period_time'], keep='first')
    duplicates_removed = initial_len - len(df)
    
    if duplicates_removed > 0:
        log.warning(f"   Removed {duplicates_removed} duplicate timestamp(s) (kept first occurrence)")
    else:
        log.info("   No duplicates found")
    
    # --- MERGE CONSTRAINTS (DAILY RESOLUTION) ---
    log.info("\nMerging constraint forecasts...")
    constraints['Date (GMT/BST)'] = pd.to_datetime(constraints['Date (GMT/BST)'])
    constraints_filtered = constraints[
        constraints['Date (GMT/BST)'].dt.date == prediction_date
    ]
    
    if len(constraints_filtered) == 0:
        log.warning(f"   No constraint data found for {prediction_date}, will try previous day")
    else:
        log.info(f"   Found constraint data for {prediction_date}")
        # Broadcast daily constraints to all 48 periods
        for col in constraints_filtered.columns:
            if col != 'Date (GMT/BST)':
                df[col] = constraints_filtered[col].iloc[0]
    
    # --- ADD BMU METADATA (SYSTEM-LEVEL) ---
    log.info("\nAdding system-level BMU metadata...")
    
    # Convert generationCapacity to numeric (handle string data)
    bmus['generationCapacity'] = pd.to_numeric(bmus['generationCapacity'], errors='coerce')
    
    total_capacity = bmus['generationCapacity'].sum()
    n_bmus = len(bmus)
    
    df['generationCapacity'] = total_capacity
    df['bmUnit'] = n_bmus
    
    log.info(f"   Total system capacity: {total_capacity:,.0f} MW")
    log.info(f"   Total BMUs: {n_bmus}")
    
    # --- ADD TEMPORAL FEATURES ---
    log.info("\nAdding temporal features...")
    df['hour'] = df['settlement_period_time'].dt.hour
    df['month'] = df['settlement_period_time'].dt.month
    
    # Add cyclical features
    df = add_cyclical_time_features(df)
    
    # --- BACKFILL MISSING VALUES FROM PREVIOUS DAY ---
    log.info("\nChecking for missing values...")
    
    # Identify columns with forecast data (exclude metadata and temporal features)
    forecast_cols = ['sys_wind_gen_forecast', 'sys_demand_forecast']
    
    # Add constraint columns dynamically
    constraint_cols = [col for col in df.columns if '_Flow' in col or '_Limit' in col]
    forecast_cols.extend(constraint_cols)
    
    # Check which columns have missing values
    missing_counts_before = df[forecast_cols].isnull().sum()
    cols_with_missing = missing_counts_before[missing_counts_before > 0]
    
    if len(cols_with_missing) > 0:
        log.warning(f"   Found missing values in {len(cols_with_missing)} column(s)")
        log.info("   Attempting to backfill from previous day's data...")
        
        # Calculate previous day's date range
        prev_day_start = prediction_start - timedelta(days=1)
        prev_day_end = prev_day_start + timedelta(days=1)
        
        # Load previous day's wind forecast
        if 'sys_wind_gen_forecast' in cols_with_missing.index:
            windfor_prev = windfor[
                (windfor['half_hour_time'] >= prev_day_start) &
                (windfor['half_hour_time'] < prev_day_end)
            ].copy()
            
            if len(windfor_prev) > 0:
                # Shift timestamps by +1 day to align with prediction date
                windfor_prev['half_hour_time'] = windfor_prev['half_hour_time'] + timedelta(days=1)
                
                # Merge and fill missing values
                df = df.merge(
                    windfor_prev[['half_hour_time', 'sys_wind_gen_forecast']].rename(
                        columns={'sys_wind_gen_forecast': 'wind_prev_day'}
                    ),
                    left_on='settlement_period_time',
                    right_on='half_hour_time',
                    how='left'
                ).drop(columns=['half_hour_time'])
                
                # Fill missing values
                df['sys_wind_gen_forecast'] = df['sys_wind_gen_forecast'].fillna(df['wind_prev_day'])
                df = df.drop(columns=['wind_prev_day'])
                
                backfilled = missing_counts_before['sys_wind_gen_forecast'] - df['sys_wind_gen_forecast'].isnull().sum()
                if backfilled > 0:
                    log.info(f"      sys_wind_gen_forecast: Backfilled {backfilled} value(s) from previous day")
        
        # Load previous day's demand forecast
        if 'sys_demand_forecast' in cols_with_missing.index:
            demandfor_prev = demandfor[
                (demandfor['half_hour_time'] >= prev_day_start) &
                (demandfor['half_hour_time'] < prev_day_end)
            ].copy()
            
            if len(demandfor_prev) > 0:
                # Shift timestamps by +1 day
                demandfor_prev['half_hour_time'] = demandfor_prev['half_hour_time'] + timedelta(days=1)
                
                df = df.merge(
                    demandfor_prev[['half_hour_time', 'sys_demand_forecast']].rename(
                        columns={'sys_demand_forecast': 'demand_prev_day'}
                    ),
                    left_on='settlement_period_time',
                    right_on='half_hour_time',
                    how='left'
                ).drop(columns=['half_hour_time'])
                
                df['sys_demand_forecast'] = df['sys_demand_forecast'].fillna(df['demand_prev_day'])
                df = df.drop(columns=['demand_prev_day'])
                
                backfilled = missing_counts_before['sys_demand_forecast'] - df['sys_demand_forecast'].isnull().sum()
                if backfilled > 0:
                    log.info(f"      sys_demand_forecast: Backfilled {backfilled} value(s) from previous day")
        
        # Load previous day's constraints
        if any(col in cols_with_missing.index for col in constraint_cols):
            constraints_prev = constraints[
                constraints['Date (GMT/BST)'].dt.date == (prediction_date - timedelta(days=1))
            ]
            
            if len(constraints_prev) > 0:
                for col in constraint_cols:
                    if col in cols_with_missing.index and col in constraints_prev.columns:
                        # Fill missing constraint values with previous day's value
                        if df[col].isnull().all():
                            df[col] = constraints_prev[col].iloc[0]
                            log.info(f"      {col}: Backfilled from previous day (all values were missing)")
    else:
        log.info("   No missing values found")
    
    # --- FINAL VALIDATION ---
    log.info("\nValidating feature completeness...")
    
    if len(df) != 48:
        log.error(f"[ERROR] Expected 48 periods, got {len(df)}")
        log.error(f"   Unique timestamps: {df['settlement_period_time'].nunique()}")
        log.error(f"   Date range: {df['settlement_period_time'].min()} to {df['settlement_period_time'].max()}")
        raise ValueError(f"Incomplete settlement periods: {len(df)}/48")
    
    # Check for required features
    missing_features = set(required_features) - set(df.columns)
    if missing_features:
        log.error(f"[ERROR] Missing required features: {missing_features}")
        raise ValueError(f"Cannot generate predictions without: {missing_features}")
    
    # Check for missing values after backfill
    missing_counts_after = df[required_features].isnull().sum()
    total_missing = missing_counts_after.sum()
    
    if total_missing > 0:
        log.info(f"\nRemaining missing values: {int(total_missing):,} ({100*total_missing/(len(df)*len(required_features)):.2f}%)")
        for col in missing_counts_after[missing_counts_after > 0].index[:5]:
            log.info(f"   {col}: {int(missing_counts_after[col])} missing")
        log.info("   XGBoost will handle remaining missing values using native support")
    else:
        log.info("   [OK] No missing values in required features")
    
    # Select only required features (in same order as training)
    feature_df = df[['settlement_period_time'] + required_features].copy()
    
    log.info(f"\n[OK] Built complete feature set: {len(feature_df)} periods x {len(required_features)} features")
    
    return feature_df

# -----------------------
# 4. Generate Predictions
# -----------------------
def generate_predictions(
    model: xgb.XGBClassifier,
    feature_df: pd.DataFrame,
    required_features: List[str],
    threshold: float
) -> pd.DataFrame:
    """Generate curtailment predictions for all settlement periods."""
    log.info("=" * 60)
    log.info("GENERATING PREDICTIONS")
    log.info("=" * 60)
    
    # Extract features for prediction (exclude settlement_period_time)
    X = feature_df[required_features]
    
    log.info(f"Running inference on {len(X)} settlement periods...")
    
    # Get probability predictions
    y_proba = model.predict_proba(X)[:, 1]
    
    # Apply threshold to get binary predictions
    y_pred = (y_proba >= threshold).astype(int)
    
    # Create results dataframe
    results = pd.DataFrame({
        'settlement_period_time': feature_df['settlement_period_time'],
        'curtailment_probability': y_proba,
        'curtailment_prediction': y_pred,
        'threshold_used': threshold
    })
    
    # Summary statistics
    n_curtailment = y_pred.sum()
    avg_prob = y_proba.mean()
    max_prob = y_proba.max()
    min_prob = y_proba.min()
    
    log.info(f"\n[OK] Predictions generated:")
    log.info(f"   Curtailment periods predicted: {n_curtailment}/48 ({100*n_curtailment/48:.1f}%)")
    log.info(f"   Average curtailment probability: {avg_prob:.4f}")
    log.info(f"   Max curtailment probability: {max_prob:.4f}")
    log.info(f"   Min curtailment probability: {min_prob:.4f}")
    
    return results

# -----------------------
# 5. Save Predictions
# -----------------------
def save_predictions(predictions: pd.DataFrame, prediction_date: datetime.date, metadata: dict):
    """Save predictions and metadata to disk."""
    log.info("=" * 60)
    log.info("SAVING PREDICTIONS")
    log.info("=" * 60)
    
    # Format date for filename (YYYYMMDD)
    date_str = prediction_date.strftime('%Y%m%d')
    
    # Save predictions CSV
    csv_path = PREDICTIONS_DIR / f"{date_str}_predictions.csv"
    predictions.to_csv(csv_path, index=False)
    log.info(f"[OK] Saved predictions to {csv_path}")
    
    # Save metadata JSON
    json_path = PREDICTIONS_DIR / f"{date_str}_metadata.json"
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    log.info(f"[OK] Saved metadata to {json_path}")
    
    # Display sample predictions
    log.info("\nSample predictions (first 5 periods):")
    log.info(predictions.head(5).to_string(index=False))
    
    log.info("\nSample predictions (last 5 periods):")
    log.info(predictions.tail(5).to_string(index=False))

# -----------------------
# 6. Main Pipeline
# -----------------------
def main():
    """Main inference pipeline."""
    log.info("\n" + "=" * 60)
    log.info("DAY-AHEAD CURTAILMENT FORECASTING - INFERENCE PIPELINE")
    log.info("=" * 60)
    
    start_time = datetime.now()
    
    try:
        # Step 1: Load model artifacts
        model, required_features, threshold = load_model_artifacts()
        
        # Step 2: Determine prediction date from forecast data
        prediction_date = get_prediction_date()
        
        # Step 3: Build features for prediction date (from forecasts only)
        feature_df = build_inference_features(prediction_date, required_features)
        
        # Step 4: Generate predictions
        predictions = generate_predictions(model, feature_df, required_features, threshold)
        
        # Step 5: Prepare metadata
        metadata = {
            "prediction_date": prediction_date.isoformat(),
            "prediction_timestamp": datetime.now().isoformat(),
            "model_file": str(MODEL_FILE.name),
            "xgboost_version": xgb.__version__,
            "threshold": float(threshold),
            "n_features": len(required_features),
            "n_periods": len(predictions),
            "n_curtailment_predicted": int(predictions['curtailment_prediction'].sum()),
            "avg_curtailment_probability": float(predictions['curtailment_probability'].mean()),
            "max_curtailment_probability": float(predictions['curtailment_probability'].max()),
            "min_curtailment_probability": float(predictions['curtailment_probability'].min()),
            "data_sources": {
                "wind_forecast": str(WINDFOR_PATH.name),
                "demand_forecast": str(DEMANDFOR_PATH.name),
                "constraints": str(CONSTRAINTS_PATH.name),
                "bmus": str(BMUS_PATH.name)
            }
        }
        
        # Step 6: Save results
        save_predictions(predictions, prediction_date, metadata)
        
        # Summary
        duration = datetime.now() - start_time
        log.info("\n" + "=" * 60)
        log.info("INFERENCE PIPELINE COMPLETED SUCCESSFULLY")
        log.info("=" * 60)
        log.info(f"Prediction date: {prediction_date}")
        log.info(f"Predictions saved: {PREDICTIONS_DIR / prediction_date.strftime('%Y%m%d')}_predictions.csv")
        log.info(f"Duration: {duration}")
        
    except Exception as e:
        log.error(f"\n[ERROR] INFERENCE PIPELINE FAILED")
        log.error(f"Error: {e}")
        import traceback
        log.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main()