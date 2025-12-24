import pandas as pd
from pathlib import Path

# Define paths
DATA_PROCESSED_DIR = Path("data/processed")
FEATURES_OUTPUT_PATH = DATA_PROCESSED_DIR / "features.parquet"

# File paths for processed data
FPN_PATH = DATA_PROCESSED_DIR / "fpn" / "fpn_cleaned.parquet"
BOALF_PATH = DATA_PROCESSED_DIR / "boalf" / "boalf_cleaned.parquet"
DEMANDFOR_PATH = DATA_PROCESSED_DIR / "demandfor" / "demandfor_cleaned.parquet"
WINDFOR_PATH = DATA_PROCESSED_DIR / "windfor" / "windfor_cleaned.parquet"
CONSTRAINTS_PATH = DATA_PROCESSED_DIR / "da_constraints" / "da_constraints_cleaned.parquet"
BMUS_PATH = DATA_PROCESSED_DIR / "bmus" / "bmus_cleaned.parquet"   # 👈 added

# ---------------------------------------------------------------------------
# 1. Load Data
# ---------------------------------------------------------------------------
def load_data() -> dict:
    """Load all required cleaned datasets into a dictionary."""
    print("Loading processed datasets...")
    return {
        "fpn": pd.read_parquet(FPN_PATH),
        "boalf": pd.read_parquet(BOALF_PATH),
        "demandfor": pd.read_parquet(DEMANDFOR_PATH),
        "windfor": pd.read_parquet(WINDFOR_PATH),
        "constraints": pd.read_parquet(CONSTRAINTS_PATH),
        "bmus": pd.read_parquet(BMUS_PATH)  # 👈 added
    }

def create_settlement_period_time(df: pd.DataFrame, settlement_col: str = 'settlementPeriod') -> pd.DataFrame:
    """
    Create settlement_period_time from settlementDate and settlement period number.
    
    Args:
        df: DataFrame with 'settlementDate' and settlement period column
        settlement_col: Name of the column containing settlement period number (1-48 or 1-50)
    
    Returns:
        DataFrame with added 'settlement_period_time' column
    """
    df = df.copy()
    
    # Ensure settlementDate is datetime
    df['settlementDate'] = pd.to_datetime(df['settlementDate'])
    
    # Convert settlement period to numeric
    df[settlement_col] = pd.to_numeric(df[settlement_col], errors='coerce')
    
    # Calculate start time of each settlement period
    # Period 1 = 00:00, Period 2 = 00:30, ..., Period 48 = 23:30
    df['minutes_offset'] = (df[settlement_col] - 1) * 30
    
    # Create settlement_period_time as the START of each period
    df['settlement_period_time'] = (
        df['settlementDate'] + pd.to_timedelta(df['minutes_offset'], unit='min')
    )
    
    # Drop temporary column
    df = df.drop(columns=['minutes_offset'], errors='ignore')
    
    return df


# ---------------------------------------------------------------------------
# 2. Target Variable Engineering
# ---------------------------------------------------------------------------
def engineer_target_variable(fpn: pd.DataFrame, boalf: pd.DataFrame) -> pd.DataFrame:
    """Engineer curtailment target variable from FPN and BOALF datasets."""
    print("Engineering target variable from FPN and BOALF data...")

    # ✅ STEP 1: Create settlement_period_time for FPN (uses single settlementPeriod)
    fpn = create_settlement_period_time(fpn, settlement_col='settlementPeriod')
    
    # ✅ STEP 2: Create settlement_period_time for BOALF (uses settlementPeriodTo)
    # We use 'To' because BOA instructions affect the END settlement period
    boalf = create_settlement_period_time(boalf, settlement_col='settlementPeriodTo')
    
    # Ensure datetime formats for time columns
    fpn['timeFrom'] = pd.to_datetime(fpn['timeFrom'])
    fpn['timeTo'] = pd.to_datetime(fpn['timeTo'])
    boalf['timeFrom'] = pd.to_datetime(boalf['timeFrom'])
    boalf['timeTo'] = pd.to_datetime(boalf['timeTo'])

    # ✅ STEP 3: Rename settlement_period_time before merge to ensure suffixes work
    fpn = fpn.rename(columns={'settlement_period_time': 'settlement_period_time_fpn'})
    boalf = boalf.rename(columns={'settlement_period_time': 'settlement_period_time_boalf'})

    # ✅ STEP 4: Merge on bmUnit, timeFrom, timeTo (keeps sub-period precision)
    features = pd.merge(
        fpn[['bmUnit', 'timeFrom', 'timeTo', 'fpn_levelFrom', 'fpn_levelTo', 'settlement_period_time_fpn']],
        boalf[['bmUnit', 'timeFrom', 'timeTo', 'boal_levelFrom', 'boal_levelTo', 'soFlag', 'settlement_period_time_boalf']],
        on=['bmUnit', 'timeFrom', 'timeTo'],
        how='outer'
    )
    
    # ✅ STEP 5: Use FPN's settlement_period_time (more reliable as it's single-period)
    # If FPN is missing, use BOALF's
    features['settlement_period_time'] = features['settlement_period_time_fpn'].fillna(
        features['settlement_period_time_boalf']
    )
    features = features.drop(columns=['settlement_period_time_fpn', 'settlement_period_time_boalf'])

    # Sort and forward fill FPN values within each bmUnit
    features = features.sort_values(['bmUnit', 'timeFrom'])
    features['fpn_levelFrom'] = features.groupby('bmUnit')['fpn_levelFrom'].ffill()
    features['fpn_levelTo'] = features.groupby('bmUnit')['fpn_levelTo'].ffill()

    # Fill missing BOALF values with FPN (no BOAL = no curtailment)
    features['boal_levelFrom'] = features['boal_levelFrom'].fillna(features['fpn_levelFrom'])
    features['boal_levelTo'] = features['boal_levelTo'].fillna(features['fpn_levelTo'])
    features['soFlag'] = features['soFlag'].fillna(False).infer_objects(copy=False)

    # ✅ STEP 6: Compute curtailment (using levelTo - final accepted level)
    features['curtailment'] = features['fpn_levelTo'] - features['boal_levelTo']
    features['curtailment'] = features['curtailment'].clip(lower=0)

    # Set curtailment to zero where soFlag is False
    features.loc[features['soFlag'] == False, 'curtailment'] = 0

    # Calculate interval duration in hours
    features['duration_h'] = (features['timeTo'] - features['timeFrom']).dt.total_seconds() / 3600
    features = features[features['duration_h'] > 0]

    # Calculate curtailment energy (MWh)
    features['curtailment_mwh'] = features['curtailment'] * features['duration_h']

    # ✅ STEP 7: Ensure settlement_period_time is timezone-naive
    features['settlement_period_time'] = pd.to_datetime(features['settlement_period_time']).dt.tz_localize(None)

    # Curtailment flag
    features['curtailment_flag'] = ((features['soFlag'] == True) & (features['curtailment'] > 0)).astype(int)

    # Drop intermediate columns
    features = features.drop(columns=[
        'timeFrom', 'timeTo', 'fpn_levelFrom', 'fpn_levelTo',
        'boal_levelFrom', 'boal_levelTo', 'soFlag', 'curtailment', 'duration_h'
    ])

    return features

# ---------------------------------------------------------------------------
# 3. Merge with Other Datasets
# ---------------------------------------------------------------------------
def merge_datasets(features: pd.DataFrame,
                   demandfor: pd.DataFrame,
                   windfor: pd.DataFrame,
                   constraints: pd.DataFrame,
                   bmus: pd.DataFrame) -> pd.DataFrame:
    """Merge engineered target features with demand forecast, wind forecast, constraints, and BMU metadata."""
    print("Merging features with demand forecast...")
    df = features.merge(
        demandfor.drop(columns=['TARGETDATE']),
        left_on='settlement_period_time',
        right_on='half_hour_time',
        how='left'
    ).drop(columns=['half_hour_time'])

    print("Merging with wind forecast...")
    df = df.merge(
        windfor,
        left_on='settlement_period_time',
        right_on='half_hour_time',
        how='left'
    ).drop(columns=['half_hour_time'])

    print("Merging with network constraints...")
    df = df.merge(
        constraints,
        left_on='settlement_period_time',
        right_on='Date (GMT/BST)',
        how='left'
    ).drop(columns=['Date (GMT/BST)'])

    print("Merging with BMU generation capacity...")
    df = df.merge(
        bmus[['bmUnit', 'generationCapacity']],
        on='bmUnit',
        how='left'
    )

    return df

# ---------------------------------------------------------------------------
# 4. Temporal Feature Engineering
# ---------------------------------------------------------------------------
def engineer_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add temporal features like hour and month based on settlement_period_time."""
    print("Engineering temporal features (hour, month)...")
    df['hour'] = df['settlement_period_time'].dt.hour
    df['month'] = df['settlement_period_time'].dt.month
    return df

# ---------------------------------------------------------------------------
# 5. Save Data
# ---------------------------------------------------------------------------
def save_features(df: pd.DataFrame, path: Path) -> None:
    """Save final engineered feature dataset."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"✅ Features saved to {path}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    data = load_data()
    features = engineer_target_variable(data["fpn"], data["boalf"])
    merged = merge_datasets(features, data["demandfor"], data["windfor"], data["constraints"], data["bmus"])  # 👈 bmus added
    merged = engineer_temporal_features(merged)
    save_features(merged, FEATURES_OUTPUT_PATH)
    print("🚀 feature_engineering.py completed successfully.")

if __name__ == "__main__":
    main()