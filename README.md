# MLOps Pipeline - Titanic Survival Prediction

**Assignment Status**:  **COMPLETED** - This is the finished assignment with complete MLOps pipeline implementation including classes, interfaces, and proper ML practices.

**Note**: This assignment was completed without using separate branches for each step - all development was done on the main branch.

## Overview

Complete MLOps pipeline for Titanic survival prediction with multiple ML models, proper data handling, and production-ready inference capabilities.

## Quick Start

```bash
# 1. Run Logistic Regression pipeline (preprocessing + training + evaluation + inference)
./scripts/run_logistic_regression.sh

# 2. Train additional models and run inference
./scripts/run_models.sh
```

## Pipeline Components

### 1. Data Preprocessing (`preprocessing.py`)
- Cleans raw Titanic data
- Handles missing values
- Creates new features (Title, FamilySize, etc.)

### 2. Feature Engineering (`featurize.py`)
- Applies transformations (scaling, encoding, binning)
- **No Data Leakage**: Fits transformers only on training data
- Saves transformers for reuse

### 3. Model Training
- **Logistic Regression** (`LogisticRegression.py`)
- **Random Forest** (`random_forest_model.py`) 
- **XGBoost** (`xgboost_model.py`)

### 4. Model Evaluation (`evaluate.py`)
- Evaluates models using proper ML practices
- Processes raw test data with saved transformers
- Generates performance metrics

### 5. Inference (`predict.py`)
- Makes predictions on new data
- Uses saved transformers for consistent preprocessing
- Saves predictions to CSV

## Scripts Usage

### Complete Pipeline
```bash
# Run Logistic Regression pipeline (preprocessing + training + evaluation + inference)
./scripts/run_logistic_regression.sh
```

### Model Comparison + Inference
```bash
# Train RandomForest + XGBoost + run inference on all models
./scripts/run_models.sh
```

### Individual Scripts
```bash
# Preprocessing
uv run python scripts/preprocessing.py \
    --input-train data/train.csv \
    --input-test data/test.csv \
    --output data/processed/processed_data.csv

# Feature Engineering
uv run python scripts/featurize.py \
    --input-frame data/processed/processed_data.csv \
    --output-train-features data/processed/train_features.csv \
    --output-train-labels data/processed/train_labels.csv \
    --output-test-features data/processed/test_features.csv \
    --output-test-labels data/processed/test_labels.csv \
    --transformers-output transformers

# Model Training
uv run python scripts/LogisticRegression.py \
    --train-features data/processed/train_features.csv \
    --train-labels data/processed/train_labels.csv \
    --model-output models/model.pkl

# Model Evaluation
uv run python scripts/evaluate.py \
    --model models/model.pkl \
    --test-features data/processed/test_features.csv \
    --test-labels data/processed/test_labels.csv \
    --transformers-dir transformers \
    --metrics-output metrics.json

# Inference
uv run python scripts/predict.py \
    --model models/model.pkl \
    --input-data data/inference_data.csv \
    --transformers-dir transformers \
    --output predictions.csv
```

## Key Features

✅ **Proper ML Practices**: No data leakage, realistic evaluation  
✅ **Multiple Models**: Logistic Regression, Random Forest, XGBoost  
✅ **Production Ready**: Saved transformers, modular design  
✅ **Complete Pipeline**: From raw data to predictions  
✅ **Object-Oriented**: Uses interfaces and classes  

## File Structure

```
mlops-2025/
├── data/
│   ├── train.csv, test.csv           # Raw data
│   ├── inference_data.csv            # Mimic dataset for inference
│   └── processed/                    # Processed data files
├── models/                           # Trained models (.pkl files)
├── transformers/                     # Saved transformers
├── scripts/                          # All pipeline scripts
│   ├── preprocessing.py             # Data cleaning
│   ├── featurize.py                 # Feature engineering
│   ├── LogisticRegression.py        # Logistic Regression training
│   ├── random_forest_model.py       # Random Forest training
│   ├── xgboost_model.py             # XGBoost training
│   ├── evaluate.py                  # Model evaluation
│   ├── predict.py                   # Inference script
│   ├── run_logistic_regression.sh   # Logistic Regression pipeline
│   └── run_models.sh                # Model comparison + inference
└── README.md                         # This file
```

## Output Files

- **Models**: `model.pkl`, `random_forest_model.pkl`, `xgboost_model.pkl`
- **Metrics**: `metrics.json`, `random_forest.json`, `xgboost.json`
- **Predictions**: `predictions_logistic.csv`, `predictions_random_forest.csv`, `predictions_xgboost.csv`

## Dependencies

- Python 3.12+
- pandas, scikit-learn, xgboost, numpy
- Managed with `uv` package manager