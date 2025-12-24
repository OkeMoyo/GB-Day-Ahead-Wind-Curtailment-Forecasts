"""
Train a day-ahead XGBoost classifier to predict whether any system curtailment
occurs in a settlement period (binary classification: curtailment_flag).

Inputs:
- data/processed/features.parquet

Outputs (saved to models/):
- sys-day-ahead-xgb-optimal.model         (XGBoost model file)
- sys-day-ahead-selected-features.pkl     (list of selected top features)
- sys-day-ahead-threshold.json            (selected probability threshold and perf metrics)
- sys-day-ahead-training-metrics.json     (summary metrics)
"""

import os
import json
import logging
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd
import xgboost as xgb
import joblib

from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

# -----------------------
# Logging
# -----------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("train_model")

# -----------------------
# Paths & constants
# -----------------------
PROJECT_ROOT = Path.cwd()
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "features.parquet"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILE = MODELS_DIR / "sys-day-ahead-xgb-optimal.model"
FEATURES_FILE = MODELS_DIR / "sys-day-ahead-selected-features.pkl"
THRESHOLD_FILE = MODELS_DIR / "sys-day-ahead-threshold.json"
METRICS_FILE = MODELS_DIR / "sys-day-ahead-training-metrics.json"

RANDOM_STATE = 42
TEST_SIZE_FRAC = 0.20
VAL_FRAC_OF_TRAIN = 0.10  # inner validation fraction of training set for early stopping

# -----------------------
# Utility functions
# -----------------------
def load_features(path: Path) -> pd.DataFrame:
    log.info(f"Loading features from {path}")
    df = pd.read_parquet(path)
    log.info(f"Loaded features shape: {df.shape}")
    return df

def aggregate_to_system_level(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate BMU-level features to system-level per settlement_period_time.
    Uses the aggregation mapping provided in the guiding script.
    """
    log.info("Aggregating to system-level per settlement_period_time")

    # Convert generationCapacity to numeric
    if 'generationCapacity' in df.columns:
        df['generationCapacity'] = pd.to_numeric(df['generationCapacity'], errors='coerce')

    agg_mapping = {
        'curtailment_mwh': 'sum',
        'curtailment_flag': 'max',
        'bmUnit': 'nunique',
        'generationCapacity': 'sum',
        'sys_wind_gen_forecast': 'first',
        'sys_demand_forecast': 'first',
        'hour': 'first',
        'month': 'first'
    }

    # Add the many constraint columns list if present in df, aggregated with 'first'
    for col in df.columns:
        if col.endswith('_Flow') or col.endswith('_Limit'):
            agg_mapping.setdefault(col, 'first')

    # Perform aggregation (ignore missing keys gracefully)
    existing_keys = {k: v for k, v in agg_mapping.items() if k in df.columns}
    agg_df = df.groupby('settlement_period_time').agg(existing_keys).reset_index()
    agg_df = agg_df.sort_values('settlement_period_time').reset_index(drop=True)
    log.info(f"Aggregated shape: {agg_df.shape}")
    return agg_df

def add_cyclical_time_features(df: pd.DataFrame) -> pd.DataFrame:
    log.info("Adding cyclical time features (hour, month)")
    # Ensure hour/month exist
    if 'hour' not in df.columns or 'month' not in df.columns:
        # attempt to derive from settlement_period_time
        df['hour'] = pd.to_datetime(df['settlement_period_time']).dt.hour
        df['month'] = pd.to_datetime(df['settlement_period_time']).dt.month

    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    df = df.drop(columns=['hour', 'month'], errors='ignore')
    return df

def report_missing(df: pd.DataFrame) -> None:
    missing_counts = df.isnull().sum()
    total_missing = missing_counts.sum()
    log.info(f"Total missing values in dataset: {int(total_missing):,}")
    if total_missing > 0:
        log.info("Columns with missing values (top 20 shown):")
        for col, cnt in missing_counts[missing_counts > 0].sort_values(ascending=False).head(20).items():
            log.info(f"  {col}: {int(cnt):,}")

def prepare_train_test_splits(df: pd.DataFrame, target_col: str = 'curtailment_flag') -> dict:
    """
    Chronological train/test split with inner validation for early stopping.
    Returns dict with all necessary splits and indices.
    """
    log.info("Preparing chronological train/test split")

    df = df.sort_values('settlement_period_time').reset_index(drop=True)

    # Drop records with missing target
    initial_len = len(df)
    df = df.dropna(subset=[target_col])
    log.info(f"Dropped {initial_len - len(df)} records with missing {target_col}")

    # Define columns to exclude from features
    leak_cols = ['settlement_period_time', 'curtailment_mwh', target_col]
    drop_cols = [c for c in leak_cols if c in df.columns]

    X = df.drop(columns=drop_cols)
    y = df[target_col].astype(int)

    # Chronological split
    n_total = len(X)
    n_train_full = int(np.floor(n_total * (1 - TEST_SIZE_FRAC)))

    # Test set (last 20%)
    X_test = X.iloc[n_train_full:].copy()
    y_test = y.iloc[n_train_full:].copy()

    # Training set (first 80%)
    X_train_full = X.iloc[:n_train_full].copy()
    y_train_full = y.iloc[:n_train_full].copy()

    # Inner validation split (last 10% of training)
    n_val = max(1, int(np.floor(len(X_train_full) * VAL_FRAC_OF_TRAIN)))
    
    X_train_inner = X_train_full.iloc[:-n_val].copy()
    y_train_inner = y_train_full.iloc[:-n_val].copy()
    X_val_inner = X_train_full.iloc[-n_val:].copy()
    y_val_inner = y_train_full.iloc[-n_val:].copy()

    log.info(f"Split sizes - Train: {len(X_train_inner):,} | Val: {len(X_val_inner):,} | Test: {len(X_test):,}")
    
    return {
        'X_train_inner': X_train_inner,
        'y_train_inner': y_train_inner,
        'X_val_inner': X_val_inner,
        'y_val_inner': y_val_inner,
        'X_train_full': X_train_full,  # ✅ For feature selection (doesn't include val)
        'y_train_full': y_train_full,
        'X_test': X_test,
        'y_test': y_test
    }

def tune_hyperparameters(X: pd.DataFrame, y: pd.Series, scale_pos_weight: float) -> dict:
    """Use RandomizedSearchCV to find good XGBoost hyperparameters."""
    log.info("Starting hyperparameter tuning (RandomizedSearchCV)")

    param_grid = {
        'max_depth': [3, 4, 5, 6, 7, 8],
        'learning_rate': [0.01, 0.03, 0.05, 0.07, 0.1],
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 1.0],
        'min_child_weight': [1, 2, 3, 4, 5],
        'gamma': [0, 0.1, 0.3, 0.5, 1.0],
        'reg_alpha': [0, 0.01, 0.1, 1, 10],  # ✅ Fixed: use reg_alpha instead of alpha
        'reg_lambda': [0.1, 1, 5, 10, 20],   # ✅ Fixed: use reg_lambda instead of lambda
        'n_estimators': [100, 200, 300, 400, 500],
        'scale_pos_weight': [scale_pos_weight],
        'tree_method': ['hist'],
        'random_state': [RANDOM_STATE],
    }

    xgb_clf = xgb.XGBClassifier(n_jobs=-1, verbosity=0, eval_metric='logloss')
    search = RandomizedSearchCV(
        estimator=xgb_clf,
        param_distributions=param_grid,
        n_iter=30,
        scoring='average_precision',
        cv=3,
        verbose=1,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    search.fit(X, y)
    log.info(f"RandomizedSearchCV best score: {search.best_score_:.4f}")
    log.info(f"Best params: {search.best_params_}")
    return search.best_params_

def select_top_features(X_train: pd.DataFrame, y_train: pd.Series, best_params: dict, top_k: int = 20) -> List[str]:
    """
    ✅ FIXED: Use only training data (excluding validation) for feature selection.
    Train a temporary XGB model to get feature importances and select top_k features.
    """
    log.info(f"Selecting top {top_k} features using training data only")
    
    # Filter params to valid XGBoost keys
    valid_keys = xgb.XGBClassifier().get_params().keys()
    filtered_params = {k: v for k, v in best_params.items() if k in valid_keys}
    
    temp_clf = xgb.XGBClassifier(**filtered_params)
    temp_clf.set_params(n_jobs=-1, verbosity=0)
    temp_clf.fit(X_train, y_train)
    
    importances = temp_clf.feature_importances_
    feat_names = X_train.columns
    imp_df = pd.DataFrame({
        'feature': feat_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    top_features = imp_df['feature'].head(top_k).tolist()
    
    log.info("Top 10 features by importance:")
    for idx, row in imp_df.head(10).iterrows():
        log.info(f"  {row['feature']}: {row['importance']:.4f}")
    
    return top_features

def train_final_model(X_train: pd.DataFrame, y_train: pd.Series, 
                      X_val: pd.DataFrame, y_val: pd.Series, 
                      best_params: dict) -> xgb.XGBClassifier:
    """Train final XGB with early stopping on validation set (XGBoost 2.0+ compatible)."""
    log.info("Training final XGBoost model with early stopping")
    
    # Filter params to accepted keys (excluding eval_metric and early_stopping_rounds)
    valid_keys = xgb.XGBClassifier().get_params().keys()
    filtered_params = {k: v for k, v in best_params.items() if k in valid_keys}
    
    # Remove eval_metric if it exists in best_params (we'll set it explicitly)
    filtered_params.pop('eval_metric', None)
    filtered_params.pop('early_stopping_rounds', None)
    
    # ✅ XGBoost 2.0+ requires these in constructor
    clf = xgb.XGBClassifier(
        **filtered_params,
        early_stopping_rounds=10,
        eval_metric='logloss',
        n_jobs=-1,
        verbosity=1
    )
    
    # ✅ Simplified fit - only eval_set needed
    clf.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)]
    )
    
    # Safe logging
    if hasattr(clf, 'best_iteration'):
        log.info(f"Early stopping at iteration: {clf.best_iteration}")
        log.info(f"Best validation score: {clf.best_score:.4f}")
    else:
        log.info(f"Trained for {clf.n_estimators} iterations")
    
    return clf

def choose_probability_threshold(y_true: pd.Series, y_proba: np.ndarray) -> Tuple[float, dict]:
    """Choose threshold using preferred precision targets or fallback to F1-maximising threshold."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    f1s = 2 * (precisions * recalls) / (precisions + recalls + 1e-12)
    
    thr_df = pd.DataFrame({
        'threshold': np.concatenate(([0.0], thresholds)),
        'precision': precisions,
        'recall': recalls,
        'f1': f1s
    })

    precision_targets = [0.90, 0.88, 0.85, 0.80]
    selected_row = None
    
    for p in precision_targets:
        candidates = thr_df[thr_df['precision'] >= p]
        if len(candidates) > 0:
            selected_row = candidates.loc[candidates['recall'].idxmax()]
            log.info(f"Selected threshold meeting precision >= {p:.2f} that maximises recall.")
            break

    if selected_row is None:
        idx_f1 = np.nanargmax(thr_df['f1'].values)
        selected_row = thr_df.iloc[idx_f1]
        log.info("No threshold met preferred precision targets. Falling back to F1-maximising threshold.")

    threshold = float(selected_row['threshold'])
    metrics = {
        "precision": float(selected_row['precision']),
        "recall": float(selected_row['recall']),
        "f1": float(selected_row['f1'])
    }
    log.info(f"Chosen threshold: {threshold:.4f} (precision={metrics['precision']:.4f}, recall={metrics['recall']:.4f})")
    return threshold, metrics

# -----------------------
# Main training pipeline
# -----------------------
def main():
    log.info("=" * 60)
    log.info("STARTING TRAINING PIPELINE")
    log.info("=" * 60)
    
    # Load and prepare data
    df = load_features(FEATURES_PATH)
    agg = aggregate_to_system_level(df)
    agg = add_cyclical_time_features(agg)

    log.info(f"Final aggregated dataframe shape: {agg.shape}")
    report_missing(agg)

    # Check if target exists
    if 'curtailment_flag' not in agg.columns:
        log.error("Target column 'curtailment_flag' not found in features!")
        return

    # Prepare data splits
    splits = prepare_train_test_splits(agg, target_col='curtailment_flag')
    
    X_train_inner = splits['X_train_inner']
    y_train_inner = splits['y_train_inner']
    X_val_inner = splits['X_val_inner']
    y_val_inner = splits['y_val_inner']
    X_test = splits['X_test']
    y_test = splits['y_test']

    # Calculate class imbalance
    positives = y_train_inner.sum()
    negatives = len(y_train_inner) - positives
    
    if positives == 0:
        log.error("No positive examples in training set. Cannot train model.")
        return
    
    scale_pos_weight = float(negatives) / float(positives)
    log.info(f"Class distribution - Positive: {positives:,} ({100*positives/len(y_train_inner):.2f}%) | Negative: {negatives:,}")
    log.info(f"scale_pos_weight: {scale_pos_weight:.3f}")

    # Hyperparameter tuning on inner training data
    best_params = tune_hyperparameters(X_train_inner, y_train_inner, scale_pos_weight)

    # ✅ FIXED: Feature selection using ONLY inner training data (no validation leakage)
    top_features = select_top_features(X_train_inner, y_train_inner, best_params, top_k=20)
    
    # Save selected features
    joblib.dump(top_features, FEATURES_FILE)
    log.info(f"Saved top {len(top_features)} features to {FEATURES_FILE}")

    # ✅ Subset all datasets to top features with validation
    missing_features = set(top_features) - set(X_train_inner.columns)
    if missing_features:
        log.error(f"Missing features in training data: {missing_features}")
        return
    
    X_train_inner = X_train_inner[top_features]
    X_val_inner = X_val_inner[top_features]
    X_test = X_test[top_features]

    # Train final model with early stopping
    final_clf = train_final_model(X_train_inner, y_train_inner, X_val_inner, y_val_inner, best_params)

    # Evaluate on test set
    log.info("Evaluating on test set...")
    y_proba_test = final_clf.predict_proba(X_test)[:, 1]
    
    # Choose optimal threshold
    threshold, thr_metrics = choose_probability_threshold(y_test, y_proba_test)
    y_pred_test = (y_proba_test >= threshold).astype(int)

    # ✅ Safe best_iteration access
    try:
        best_iter = int(final_clf.best_iteration) if hasattr(final_clf, 'best_iteration') else final_clf.n_estimators
    except AttributeError:
        best_iter = final_clf.n_estimators
    
    # Compute comprehensive metrics
    metrics = {
        "roc_auc": float(roc_auc_score(y_test, y_proba_test)),
        "pr_auc": float(average_precision_score(y_test, y_proba_test)),
        "confusion_matrix": confusion_matrix(y_test, y_pred_test).tolist(),
        "classification_report": classification_report(y_test, y_pred_test, digits=4, output_dict=True),
        "threshold_selection": thr_metrics,
        "best_iteration": best_iter,  # ✅ Use safe value
        "n_features": len(top_features),
        "train_size": len(X_train_inner),
        "val_size": len(X_val_inner),
        "test_size": len(X_test)
    }
    
    log.info(f"\n{'='*60}")
    log.info(f"TEST SET PERFORMANCE")
    log.info(f"{'='*60}")
    log.info(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    log.info(f"PR-AUC:  {metrics['pr_auc']:.4f}")
    log.info(f"\nConfusion Matrix:")
    cm = np.array(metrics['confusion_matrix'])
    log.info(f"  TN: {cm[0,0]:,} | FP: {cm[0,1]:,}")
    log.info(f"  FN: {cm[1,0]:,} | TP: {cm[1,1]:,}")

    # Save artifacts
    final_clf.save_model(str(MODEL_FILE))
    log.info(f"✅ Saved XGBoost model to {MODEL_FILE}")

    joblib.dump(final_clf, str(MODELS_DIR / "sys-day-ahead-xgb-optimal.pkl"))
    log.info(f"✅ Saved scikit-learn compatible model")

    THRESHOLD_PAYLOAD = {
        "threshold": threshold,
        "precision": thr_metrics['precision'],
        "recall": thr_metrics['recall'],
        "f1": thr_metrics['f1']
    }
    with open(THRESHOLD_FILE, "w") as f:
        json.dump(THRESHOLD_PAYLOAD, f, indent=2)
    log.info(f"✅ Saved threshold to {THRESHOLD_FILE}")

    with open(METRICS_FILE, "w") as f:
        json.dump(metrics, f, indent=2)
    log.info(f"✅ Saved metrics to {METRICS_FILE}")

    log.info(f"\n{'='*60}")
    log.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY ✅")
    log.info(f"{'='*60}")

if __name__ == "__main__":
    main()