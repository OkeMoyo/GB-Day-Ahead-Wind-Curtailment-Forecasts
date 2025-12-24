import pandas as pd
from pathlib import Path

RAW_DATA_PATH = Path("data/raw/da_constraints/da_constraints.parquet")
PROCESSED_DATA_PATH = Path("data/processed/da_constraints/da_constraints_cleaned.parquet")

def load_data(path: Path) -> pd.DataFrame:
    """Load raw day-ahead constraints CSV data."""
    return pd.read_parquet(path)

def clean_constraints(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and transform day-ahead constraints data."""
    
    # Pivot so each Constraint Group has separate columns for Limit and Flow
    df_pivot = df.pivot_table(
        index=['Date (GMT/BST)'],
        columns='Constraint Group',
        values=['Limit (MW)', 'Flow (MW)']
    )

    # Flatten MultiIndex columns
    df_pivot.columns = [
        f"{col[1]}_{col[0].replace(' (MW)', '').replace(' ', '_')}"
        for col in df_pivot.columns
    ]

    df_pivot = df_pivot.reset_index()

    # Drop columns with 100% missing values (e.g., GETEX_Flow and GETEX_Limit)
    df_pivot = df_pivot.drop(columns=['GETEX_Flow', 'GETEX_Limit'], errors='ignore')

    return df_pivot

def save_data(df: pd.DataFrame, path: Path) -> None:
    """Save cleaned constraints data as a Parquet file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)

def main():
    print(f"Loading data from {RAW_DATA_PATH}...")
    constraints_raw = load_data(RAW_DATA_PATH)

    print("Cleaning constraints data...")
    constraints_cleaned = clean_constraints(constraints_raw)

    print(f"Saving cleaned data to {PROCESSED_DATA_PATH}...")
    save_data(constraints_cleaned, PROCESSED_DATA_PATH)

    print("✅ clean_da_constraints.py completed successfully.")

if __name__ == "__main__":
    main()