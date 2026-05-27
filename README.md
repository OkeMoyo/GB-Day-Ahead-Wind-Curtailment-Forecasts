# Day-Ahead Wind Curtailment Forecasts Dashboard
This project implements an MLOps process to generate day-ahead wind curtailment forecasts.

# Python 3.14 Setup
Use the Conda-forge environment for Python 3.14 to avoid source builds for the scientific stack.

Recommended solver: `libmamba`. If you are installing Conda from scratch on macOS, Miniforge plus the `conda-libmamba-solver` package is the most reliable setup.

```bash
conda install -n base conda-libmamba-solver
conda config --set solver libmamba
conda env create -f environment.yml
conda activate gb-day-ahead-wind-curtailment-forecasts
conda env update -n gb-day-ahead-wind-curtailment-forecasts -f environment-extras.yml
```

The `requirements.txt` file is still available for legacy pip-based installs, but the Conda-forge path is the preferred setup for Python 3.14 on macOS.

# Pipeline
## Data ingestion
Data is ingested from various API
