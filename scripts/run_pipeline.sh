#!/bin/bash

# MLOps Pipeline Runner Script
# This script runs the complete machine learning pipeline for the Titanic dataset

set -e  # Exit on any error

echo "Starting MLOps Pipeline for Titanic Dataset"
echo "=========================================="

# Create necessary directories
echo "Creating necessary directories..."
mkdir -p data/processed
mkdir -p models

# Step 1: Preprocessing
echo ""
echo "Step 1: Preprocessing data..."
uv run python scripts/preprocessing.py \
    --input-train data/train.csv \
    --input-test data/test.csv \
    --output data/processed/processed_data.csv

echo "Preprocessing completed!"

# Step 2: Feature Engineering
echo ""
echo "Step 2: Feature engineering..."
uv run python scripts/featurize.py \
    --input-frame data/processed/processed_data.csv \
    --output-train-features data/processed/train_features.csv \
    --output-train-labels data/processed/train_labels.csv \
    --output-test-features data/processed/test_features.csv \
    --output-test-labels data/processed/test_labels.csv \
    --transformers-output transformers

echo "Feature engineering completed!"

# Step 3: Model Training
echo ""
echo "Step 3: Training model..."
uv run python scripts/models.py \
    --train-features data/processed/train_features.csv \
    --train-labels data/processed/train_labels.csv \
    --model-output models/model.pkl

echo "Model training completed!"

# Step 4: Model Evaluation
echo ""
echo "Step 4: Evaluating model..."
uv run python scripts/evaluate.py \
    --model models/model.pkl \
    --test-features data/processed/test_features.csv \
    --test-labels data/processed/test_labels.csv \
    --transformers-dir transformers \
    --metrics-output metrics.json

echo "Model evaluation completed!"

echo ""
echo "Pipeline completed successfully!"
echo "==============================="
echo "Model Performance Metrics:"
cat metrics.json
echo ""
echo "Generated files:"
echo "  - data/processed/processed_data.csv (preprocessed data)"
echo "  - data/processed/train_features.csv (training features)"
echo "  - data/processed/train_labels.csv (training labels)"
echo "  - data/processed/test_features.csv (test features)"
echo "  - data/processed/test_labels.csv (test labels)"
echo "  - models/model.pkl (trained model)"
echo "  - metrics.json (evaluation metrics)"
echo ""
echo "Ready for deployment!"
