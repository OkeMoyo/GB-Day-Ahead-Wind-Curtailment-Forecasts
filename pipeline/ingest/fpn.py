import os
import glob
import time
import logging
import requests
import pandas as pd
from datetime import datetime, time, timedelta

# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------
DATA_DIR = "data/raw/fpn"
OUTPUT_FILE = "data/raw/fpn/fpn.parquet"
BMUS_FILE = "data/raw/bmus/bmUnits.parquet"

# Configurable date range
START_DATE = datetime(2024, 1, 1, 0, 0)
yesterday = datetime.today().date() - timedelta(days=1)
END_DATE = datetime.combine(yesterday, time(23, 30))

CHUNK_SIZE_DAYS = 7
MAX_RETRIES = 3
BACKOFF_FACTOR = 1  # faster retries

API_URL = "https://data.elexon.co.uk/bmrs/api/v1/balancing/physical"
DATASET = "PN"

# -------------------------------------------------------------------
# Logging setup
# -------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

# -------------------------------------------------------------------
# Functions
# -------------------------------------------------------------------
def fetch_fpn_for_bmu(bmu_id, start_date, end_date, chunk_size_days=7):
    """Fetch FPN data for a single BMU in date chunks with retry logic."""
    all_chunks = []
    start = datetime.fromisoformat(start_date.replace("Z", "+00:00"))
    end = datetime.fromisoformat(end_date.replace("Z", "+00:00"))

    chunk_start = start
    while chunk_start < end:
        chunk_end = min(chunk_start + timedelta(days=chunk_size_days), end)
        params = {
            "dataset": DATASET,
            "bmUnit": bmu_id,
            "from": chunk_start.strftime("%Y-%m-%dT%H:%MZ"),
            "to": chunk_end.strftime("%Y-%m-%dT%H:%MZ"),
        }

        retries = 0
        while retries < MAX_RETRIES:
            try:
                response = requests.get(API_URL, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()

                df = pd.DataFrame(data.get("data", []))
                if not df.empty:
                    all_chunks.append(df)

                break  # exit retry loop if successful
            except Exception as e:
                retries += 1
                wait_time = BACKOFF_FACTOR * (2 ** (retries - 1))
                logging.warning(
                    f"Retry {retries}/{MAX_RETRIES} for BMU {bmu_id} "
                    f"chunk {chunk_start} → {chunk_end} due to error: {e}"
                )
                time.sleep(wait_time)

        chunk_start = chunk_end

    return pd.concat(all_chunks, ignore_index=True) if all_chunks else pd.DataFrame()


def save_intermediate(df, bmu_id, output_dir):
    """Save BMU dataframe to intermediate parquet."""
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"{bmu_id}.parquet")
    df.to_parquet(file_path, index=False)


def merge_intermediates(output_dir, final_output):
    """Merge all intermediate parquet files and save as final parquet."""
    files = glob.glob(os.path.join(output_dir, "*.parquet"))
    if not files:
        logging.warning("No intermediate files found to merge.")
        return

    df_list = [pd.read_parquet(f) for f in files if os.path.getsize(f) > 0]
    if df_list:
        merged = pd.concat(df_list, ignore_index=True)
        merged.to_parquet(final_output, index=False)
        logging.info(f"Merged {len(df_list)} BMU files into {final_output}")
    else:
        logging.warning("No non-empty intermediate files to merge.")

    # Cleanup
    for f in files:
        os.remove(f)
    logging.info("Intermediate files deleted after merge.")


def run_extraction():
    """Main extraction pipeline for FPN data."""
    # Load BMUs
    bmus = pd.read_parquet(BMUS_FILE)
    wind_bmus = bmus[(bmus["fuelType"] == "WIND") & bmus["elexonBmUnit"].notna()]
    wind_bmus = wind_bmus[wind_bmus["elexonBmUnit"] != "None"]

    bm_units = wind_bmus["elexonBmUnit"].unique().tolist()
    logging.info(f"Found {len(bm_units)} valid wind BMUs to process.")

    total = len(bm_units)
    for i, bmu_id in enumerate(bm_units, start=1):
        logging.info(f"({i}/{total}) Fetching data for BMU: {bmu_id}")
        df = fetch_fpn_for_bmu(bmu_id, START_DATE, END_DATE, CHUNK_SIZE_DAYS)
        if not df.empty:
            save_intermediate(df, bmu_id, DATA_DIR)
        else:
            logging.warning(f"No data retrieved for BMU {bmu_id}")

    merge_intermediates(DATA_DIR, OUTPUT_FILE)


# -------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------
if __name__ == "__main__":
    run_extraction()