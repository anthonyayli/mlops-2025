# MLOps Pipeline - Titanic Survival Prediction

This repository contains a complete MLOps pipeline for predicting Titanic passenger survival using multiple machine learning models.

## Pipeline Overview

The pipeline consists of several stages:
1. **Data Preprocessing** - Clean and prepare raw data
2. **Feature Engineering** - Transform features and create train/test splits
3. **Model Training** - Train multiple ML models
4. **Model Evaluation** - Evaluate models using proper ML practices

## Scripts Documentation

### 1. Data Preprocessing (`scripts/preprocessing.py`)

**Purpose**: Clean and prepare the raw Titanic dataset

**Input**:
- `--input-train`: Path to training data CSV file (e.g., `data/train.csv`)
- `--input-test`: Path to test data CSV file (e.g., `data/test.csv`)
- `--output`: Path to save processed data (e.g., `data/processed/processed_data.csv`)

**Output**:
- Combined and cleaned dataset saved as CSV
- Handles missing values, creates new features (Title, Family_size)
- Removes unnecessary columns (Cabin, PassengerId)

**Example**:
```bash
uv run python scripts/preprocessing.py \
    --input-train data/train.csv \
    --input-test data/test.csv \
    --output data/processed/processed_data.csv
```

### 2. Feature Engineering (`scripts/featurize.py`)

**Purpose**: Transform features and create train/test splits with proper ML practices

**Input**:
- `--input-frame`: Path to processed data CSV (e.g., `data/processed/processed_data.csv`)
- `--output-train-features`: Path to save training features (e.g., `data/processed/train_features.csv`)
- `--output-train-labels`: Path to save training labels (e.g., `data/processed/train_labels.csv`)
- `--output-test-features`: Path to save raw test features (e.g., `data/processed/test_features.csv`)
- `--output-test-labels`: Path to save test labels (e.g., `data/processed/test_labels.csv`)
- `--transformers-output`: Directory to save fitted transformers (required)

**Output**:
- Transformed training features (scaled, encoded, binned)
- Raw test features (not transformed - for realistic evaluation)
- Training and test labels
- Saved transformers (`num_cat_transformer.pkl`, `bins_transformer.pkl`)

**Example**:
```bash
uv run python scripts/featurize.py \
    --input-frame data/processed/processed_data.csv \
    --output-train-features data/processed/train_features.csv \
    --output-train-labels data/processed/train_labels.csv \
    --output-test-features data/processed/test_features.csv \
    --output-test-labels data/processed/test_labels.csv \
    --transformers-output transformers
```

### 3. Model Training Scripts

#### 3.1 Logistic Regression (`scripts/models.py`)

**Purpose**: Train a Logistic Regression model

**Input**:
- `--train-features`: Path to transformed training features CSV
- `--train-labels`: Path to training labels CSV
- `--model-output`: Path to save trained model (default: `models/model.pkl`)

**Output**:
- Trained Logistic Regression model saved as pickle file

**Example**:
```bash
uv run python scripts/models.py \
    --train-features data/processed/train_features.csv \
    --train-labels data/processed/train_labels.csv \
    --model-output models/model.pkl
```

#### 3.2 Random Forest (`scripts/random_forest_model.py`)

**Purpose**: Train a Random Forest model

**Input**:
- `--train-features`: Path to transformed training features CSV
- `--train-labels`: Path to training labels CSV
- `--model-output`: Path to save trained model (default: `models/random_forest_model.pkl`)

**Output**:
- Trained Random Forest model saved as pickle file

**Example**:
```bash
uv run python scripts/random_forest_model.py \
    --train-features data/processed/train_features.csv \
    --train-labels data/processed/train_labels.csv \
    --model-output models/random_forest_model.pkl
```

#### 3.3 XGBoost (`scripts/xgboost_model.py`)

**Purpose**: Train an XGBoost model

**Input**:
- `--train-features`: Path to transformed training features CSV
- `--train-labels`: Path to training labels CSV
- `--model-output`: Path to save trained model (default: `models/xgboost_model.pkl`)

**Output**:
- Trained XGBoost model saved as pickle file

**Example**:
```bash
uv run python scripts/xgboost_model.py \
    --train-features data/processed/train_features.csv \
    --train-labels data/processed/train_labels.csv \
    --model-output models/xgboost_model.pkl
```

### 4. Model Evaluation (`scripts/evaluate.py`)

**Purpose**: Evaluate trained models using proper ML practices (process raw test data)

**Input**:
- `--model`: Path to trained model pickle file
- `--test-features`: Path to raw test features CSV (not transformed)
- `--test-labels`: Path to test labels CSV
- `--transformers-dir`: Directory containing saved transformers (default: `transformers`)
- `--metrics-output`: Optional path to save metrics as JSON

**Output**:
- Model performance metrics (accuracy, precision, recall, f1_score)
- Optional JSON file with metrics

**Example**:
```bash
uv run python scripts/evaluate.py \
    --model models/model.pkl \
    --test-features data/processed/test_features.csv \
    --test-labels data/processed/test_labels.csv \
    --transformers-dir transformers \
    --metrics-output metrics.json
```

## Pipeline Scripts

### 1. Complete Pipeline (`scripts/run_pipeline.sh`)

**Purpose**: Run the complete preprocessing and training pipeline

**Input**: Raw data files (`data/train.csv`, `data/test.csv`)

**Output**:
- Processed data files
- Trained Logistic Regression model
- Model evaluation metrics
- Saved transformers

**Usage**:
```bash
./scripts/run_pipeline.sh
```

### 2. Model Comparison (`scripts/run_models.sh`)

**Purpose**: Train and evaluate RandomForest and XGBoost models

**Prerequisites**: Must run `run_pipeline.sh` first

**Input**: Processed data from pipeline

**Output**:
- Trained RandomForest and XGBoost models
- Evaluation metrics for both models (`random_forest.json`, `xgboost.json`)

**Usage**:
```bash
./scripts/run_models.sh
```

## File Structure

```
mlops-2025/
├── data/
│   ├── train.csv                    # Raw training data
│   ├── test.csv                     # Raw test data
│   └── processed/
│       ├── processed_data.csv       # Cleaned combined data
│       ├── train_features.csv       # Transformed training features
│       ├── train_labels.csv         # Training labels
│       ├── test_features.csv        # Raw test features (not transformed)
│       └── test_labels.csv          # Test labels
├── models/
│   ├── model.pkl                    # Logistic Regression model
│   ├── random_forest_model.pkl      # Random Forest model
│   └── xgboost_model.pkl            # XGBoost model
├── transformers/
│   ├── num_cat_transformer.pkl      # Scaling and encoding transformer
│   └── bins_transformer.pkl         # Binning transformer
├── scripts/
│   ├── preprocessing.py             # Data cleaning
│   ├── featurize.py                 # Feature engineering
│   ├── models.py                    # Logistic Regression training
│   ├── random_forest_model.py       # Random Forest training
│   ├── xgboost_model.py             # XGBoost training
│   ├── evaluate.py                  # Model evaluation
│   ├── run_pipeline.sh              # Complete pipeline
│   └── run_models.sh                # Model comparison
└── README.md                        # This file
```

## Key Features

### Proper ML Practices
- **No Data Leakage**: Transformers are fitted only on training data
- **Realistic Evaluation**: Test data is processed during evaluation using saved transformers
- **Consistent Preprocessing**: Same transformations applied to training and test data

### Model Comparison
- **Multiple Algorithms**: Logistic Regression, Random Forest, XGBoost
- **Consistent Evaluation**: All models evaluated using the same methodology
- **Performance Metrics**: Accuracy, Precision, Recall, F1-Score

### Production Ready
- **Saved Transformers**: Can be reused for new data
- **Modular Design**: Each script has a specific purpose
- **Error Handling**: Proper error handling and validation

## Dependencies

- Python 3.12+
- pandas
- scikit-learn
- xgboost
- numpy

## Quick Start

1. **Run the complete pipeline**:
   ```bash
   ./scripts/run_pipeline.sh
   ```

2. **Train and compare additional models**:
   ```bash
   ./scripts/run_models.sh
   ```

3. **Check results**:
   - Model files in `models/` directory
   - Metrics in JSON files
   - Transformers in `transformers/` directory

## Notes

- The pipeline follows proper ML practices with no data leakage
- Test data is processed during evaluation using saved transformers
- All models are trained on the same preprocessed training data
- Transformers can be reused for new data in production