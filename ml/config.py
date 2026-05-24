"""
ML Pipeline Configuration
Author: Bharath Krishna
Project: Dr. Friend
"""

import os
from pathlib import Path

# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ML_DIR = PROJECT_ROOT / "ml"
MODELS_DIR = ML_DIR / "models"
ARTIFACTS_DIR = ML_DIR / "artifacts"

# Dataset
DATASET_PATH = DATA_DIR / "dr_friend_dataset.csv"

# ============================================================================
# MODEL PARAMETERS
# ============================================================================

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Target variable
TARGET_COL = 'disease'

# Feature columns (will be populated from dataset)
VITAL_COLS = [
    'temperature_c', 'heart_rate_bpm', 'systolic_bp', 'diastolic_bp',
    'pulse_pressure', 'spo2_percent', 'respiratory_rate', 'blood_glucose_mgdl'
]

DEMOGRAPHIC_COLS = ['age', 'gender', 'height_cm', 'weight_kg', 'bmi']

# ============================================================================
# MLFLOW CONFIGURATION
# ============================================================================

MLFLOW_TRACKING_URI = "file:./mlruns"
EXPERIMENT_NAME = "dr_friend_disease_prediction"

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

# Age groups for binning
AGE_BINS = [0, 18, 45, 65, 100]
AGE_LABELS = ['child', 'adult', 'middle_age', 'elderly']

# Temperature categories
TEMP_BINS = [0, 37.5, 38.5, 100]
TEMP_LABELS = ['normal', 'low_grade_fever', 'high_grade_fever']

# ============================================================================
# HYPERPARAMETER SEARCH SPACES
# ============================================================================

# Logistic Regression (baseline)
LOGISTIC_PARAMS = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga'],
    'max_iter': [1000]
}


# XGBoost
XGBOOST_PARAMS = {
    'max_depth': [3, 5, 7, 9],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'n_estimators': [100, 200, 300],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0]
}

# Random Forest
RF_PARAMS = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# LightGBM
LGBM_PARAMS = {
    'num_leaves': [31, 50, 70],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 200, 300],
    'max_depth': [-1, 10, 20]
}

# ============================================================================
# EVALUATION METRICS
# ============================================================================

TARGET_F1_SCORE = 0.75  # Minimum target for mid-sem

print(f"Configuration loaded successfully")
print(f"Project root: {PROJECT_ROOT}")
print(f"Dataset path: {DATASET_PATH}")
print(f"Target F1 score: {TARGET_F1_SCORE}")