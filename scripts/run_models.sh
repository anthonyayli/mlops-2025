#!/bin/bash

# Model Training and Evaluation Script
# This script trains RandomForest and XGBoost models and evaluates them

set -e  # Exit on any error

echo "Starting Model Training and Evaluation"
echo "====================================="

# Ensure we have the processed data
if [ ! -f "data/processed/train_features.csv" ]; then
    echo "Error: Processed data not found. Please run the pipeline first."
    exit 1
fi

# Train RandomForest Model
echo ""
echo "Training RandomForest model..."
uv run python scripts/random_forest_model.py \
    --train-features data/processed/train_features.csv \
    --train-labels data/processed/train_labels.csv \
    --model-output models/random_forest_model.pkl

echo "Evaluating RandomForest model..."
uv run python scripts/evaluate.py \
    --model models/random_forest_model.pkl \
    --test-features data/processed/test_features.csv \
    --test-labels data/processed/test_labels.csv \
    --metrics-output random_forest.json

echo "RandomForest evaluation completed!"

# Train XGBoost Model
echo ""
echo "Training XGBoost model..."
uv run python scripts/xgboost_model.py \
    --train-features data/processed/train_features.csv \
    --train-labels data/processed/train_labels.csv \
    --model-output models/xgboost_model.pkl

echo "Evaluating XGBoost model..."
uv run python scripts/evaluate.py \
    --model models/xgboost_model.pkl \
    --test-features data/processed/test_features.csv \
    --test-labels data/processed/test_labels.csv \
    --metrics-output xgboost.json

echo "XGBoost evaluation completed!"

echo ""
echo "All models trained and evaluated successfully!"
echo "============================================="
echo "Results saved to:"
echo "  - random_forest.json (RandomForest metrics)"
echo "  - xgboost.json (XGBoost metrics)"
echo ""
echo "Model files saved to:"
echo "  - models/random_forest_model.pkl"
echo "  - models/xgboost_model.pkl"
