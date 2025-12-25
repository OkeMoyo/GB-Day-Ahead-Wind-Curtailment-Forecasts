@echo off
echo ============================================================
echo INITIAL DATA SETUP - FULL HISTORICAL DOWNLOAD
echo ============================================================

cd /d "C:\Users\Oke\Documents\Personal\MOBI Analytics\Wind Curtailment Forecasting"
call .venv\Scripts\activate

echo.
echo [1/5] Downloading BMU list...
python -m pipeline.ingest.bmus

echo.
echo [2/5] Downloading full DA constraints history...
python -m pipeline.ingest.da_constraints

echo.
echo [3/5] Running incremental extraction for other datasets...
python -m pipeline.ingest.extract_incremental

echo.
echo [4/5] Cleaning all data...
python -m pipeline.preprocessing.clean_windfor
python -m pipeline.preprocessing.clean_demandfor
python -m pipeline.preprocessing.clean_da_constraints
python -m pipeline.preprocessing.clean_bmus

echo.
echo [5/5] Generating initial predictions...
python -m pipeline.inference

echo.
echo ============================================================
echo SETUP COMPLETED SUCCESSFULLY
echo ============================================================
pause