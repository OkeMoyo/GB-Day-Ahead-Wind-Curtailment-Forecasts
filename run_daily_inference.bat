@echo off
echo ============================================================
echo DAILY CURTAILMENT FORECAST UPDATE
echo Started: %date% %time%
echo ============================================================

cd /d "C:\Users\Oke\Documents\Personal\MOBI Analytics\Wind Curtailment Forecasting"

REM Activate virtual environment
call .venv\Scripts\activate

REM Step 0: Sync with GitHub (prevent merge conflicts)
echo.
echo [0/5] Syncing with GitHub...
git fetch origin
git pull origin main --no-edit
if errorlevel 1 (
    echo WARNING: Git pull failed, continuing anyway...
)

REM Step 1: Extract latest forecast data
echo.
echo [1/5] Extracting latest forecast data...
python -m pipeline.ingest.extract_incremental
if errorlevel 1 (
    echo ERROR: Data extraction failed!
    pause
    exit /b 1
)

REM Step 2: Clean forecast data
echo.
echo [2/5] Cleaning forecast data...
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
echo [3/5] Generating predictions...
python -m pipeline.inference
if errorlevel 1 (
    echo ERROR: Inference failed!
    pause
    exit /b 1
)

REM Step 4: Stage changes
echo.
echo [4/5] Staging changes...
git add predictions/
git add data/raw/ 2>nul
git add data/processed/ 2>nul

REM Step 5: Commit and push
echo.
echo [5/5] Committing and pushing to GitHub...

REM Check if there are changes to commit
git diff-index --quiet HEAD --
if errorlevel 1 (
    echo Changes detected, committing...
    git commit -m "Auto-update: Predictions for %date% %time%"
    
    REM Push with retry logic
    git push origin main
    if errorlevel 1 (
        echo WARNING: First push attempt failed, retrying...
        timeout /t 3 /nobreak >nul
        git pull origin main --no-edit
        git push origin main
        if errorlevel 1 (
            echo ERROR: Git push failed after retry!
            pause
            exit /b 1
        )
    )
    echo Successfully pushed to GitHub
) else (
    echo No changes to commit (predictions unchanged)
)

echo.
echo ============================================================
echo COMPLETED SUCCESSFULLY at %date% %time%
echo ============================================================

REM Deactivate virtual environment
deactivate