"""
Configuration file for traffic flow prediction project
Centralized settings for data paths, model parameters, and feature engineering
"""

import os
from pathlib import Path

# Project root directory
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"

# Ensure directories exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, RESULTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Data settings
TARGET_COLUMN = "traffic_flow"  # Target variable for prediction
TIME_COLUMN = "timestamp"  # Time column name
FEATURE_COLUMNS = [
    "hour", "day_of_week", "month", "is_weekend", 
    "is_peak_hour", "is_holiday", "day_of_year"
]

# Peak hour definitions (24-hour format)
MORNING_PEAK_START = 7
MORNING_PEAK_END = 9
EVENING_PEAK_START = 17
EVENING_PEAK_END = 19

# Holiday list (format: 'MM-DD')
HOLIDAYS = [
    '01-01',  # New Year's Day
    '02-14',  # Valentine's Day
    '04-04',  # Qingming Festival (approximate)
    '05-01',  # Labor Day
    '06-10',  # Dragon Boat Festival (approximate)
    '09-21',  # Mid-Autumn Festival (approximate)
    '10-01',  # National Day
    '10-10',  # Double Tenth Day
    '12-25',  # Christmas
]

# Data split ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Model parameters
MODEL_PARAMS = {
    'xgboost': {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42
    },
    'random_forest': {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'random_state': 42,
        'n_jobs': -1
    },
    'svm': {
        'kernel': 'rbf',
        'C': 1.0,
        'gamma': 'scale',
        'epsilon': 0.1
    }
}

# Evaluation metrics
METRICS = ['MAE', 'RMSE', 'MAPE', 'R2']

# Random seed for reproducibility
RANDOM_SEED = 42
