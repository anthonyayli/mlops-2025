#!/bin/bash

# Complete Model Pipeline Script
# This script trains RandomForest and XGBoost models, evaluates them, and runs inference

set -e  # Exit on any error

echo "Starting Complete Model Pipeline"
echo "==============================="


if [ ! -f "data/processed/train_features.csv" ]; then
    echo "Error: Processed data not found. Please run the pipeline first."
    exit 1
fi


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
    --transformers-dir transformers \
    --metrics-output random_forest.json

echo "RandomForest evaluation completed!"


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
    --transformers-dir transformers \
    --metrics-output xgboost.json

echo "XGBoost evaluation completed!"

echo ""
echo "Running inference on new data..."
echo "==============================="

# Check if inference data exists
if [ ! -f "data/inference_data.csv" ]; then
    echo "Error: inference_data.csv not found. Please create it first."
    exit 1
fi

echo ""
echo "Making predictions with Logistic Regression model..."
uv run python scripts/predict.py \
    --model models/model.pkl \
    --input-data data/inference_data.csv \
    --transformers-dir transformers \
    --output predictions_logistic.csv

echo ""
echo "Making predictions with RandomForest model..."
uv run python scripts/predict.py \
    --model models/random_forest_model.pkl \
    --input-data data/inference_data.csv \
    --transformers-dir transformers \
    --output predictions_random_forest.csv

echo ""
echo "Making predictions with XGBoost model..."
uv run python scripts/predict.py \
    --model models/xgboost_model.pkl \
    --input-data data/inference_data.csv \
    --transformers-dir transformers \
    --output predictions_xgboost.csv

echo ""
echo "All models trained, evaluated, and inference completed successfully!"
echo "===================================================================="
echo "Evaluation results saved to:"
echo "  - random_forest.json (RandomForest metrics)"
echo "  - xgboost.json (XGBoost metrics)"
echo ""
echo "Model files saved to:"
echo "  - models/model.pkl (Logistic Regression)"
echo "  - models/random_forest_model.pkl"
echo "  - models/xgboost_model.pkl"
echo ""
echo "Inference predictions saved to:"
echo "  - predictions_logistic.csv"
echo "  - predictions_random_forest.csv"
echo "  - predictions_xgboost.csv"
