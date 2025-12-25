@echo off
echo ============================================================
echo DAILY CURTAILMENT FORECAST UPDATE
echo Started: %date% %time%
echo ============================================================

cd /d "C:\Users\Oke\Documents\Personal\MOBI Analytics\Wind Curtailment Forecasting"

REM Activate virtual environment
call .venv\Scripts\activate

REM Step 1: Extract latest forecast data
echo.
echo [1/4] Extracting latest forecast data...
python -m pipeline.ingest.extract_incremental
if errorlevel 1 (
    echo ERROR: Data extraction failed!
    pause
    exit /b 1
)

REM Step 2: Clean forecast data
echo.
echo [2/4] Cleaning forecast data...
python -m pipeline.preprocessing.clean_windfor
python -m pipeline.preprocessing.clean_demandfor
python -m pipeline.preprocessing.clean_da_constraints
python -m pipeline.preprocessing.clean_bmus

if errorlevel 1 (
    echo ERROR: Data cleaning failed!
    pause
    exit /b 1
)

REM Step 3: Generate predictions (builds features internally)
echo.
echo [3/4] Generating predictions...
python -m pipeline.inference
if errorlevel 1 (
    echo ERROR: Inference failed!
    pause
    exit /b 1
)

REM Step 4: Push to GitHub for Streamlit Cloud
echo.
echo [4/4] Pushing to GitHub...
git add predictions/

REM Check if there are changes to commit
git diff-index --quiet HEAD --
if errorlevel 1 (
    echo Changes detected, committing...
    git commit -m "Auto-update: Predictions for %date% %time%"
    git push origin main
    
    if errorlevel 1 (
        echo WARNING: Git push failed (check internet connection)
    ) else (
        echo Successfully pushed to GitHub
    )
) else (
    echo No changes to commit (predictions unchanged)
)

echo.
echo ============================================================
echo COMPLETED SUCCESSFULLY at %date% %time%
echo ============================================================

REM Deactivate virtual environment
deactivate