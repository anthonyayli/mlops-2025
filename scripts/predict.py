#!/usr/bin/env python3
"""
Inference script for Titanic survival prediction.
Loads a trained model and applies it to new data for predictions.
"""

import argparse
import pandas as pd
import pickle
import os
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    """Build command line argument parser."""
    p = argparse.ArgumentParser(description="Make predictions on new Titanic data")
    p.add_argument("--model", required=True, help="Path to trained model file (.pkl)")
    p.add_argument("--input-data", required=True, help="Path to inference data CSV file")
    p.add_argument("--transformers-dir", required=True, help="Directory containing saved transformers")
    p.add_argument("--output", required=True, help="Path to save predictions CSV file")
    return p


def load_model(model_path: str):
    """Load trained model from pickle file."""
    print(f"Loading model from {model_path}...")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print(f"Model loaded successfully: {type(model).__name__}")
    return model


def preprocess_inference_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply preprocessing to inference data.
    This mirrors the preprocessing logic from preprocessing.py
    """
    print("Applying preprocessing to inference data...")
    
    # Create a copy to avoid modifying original
    df_processed = df.copy()
    
    # Handle missing values in Age
    df_processed['Age'].fillna(df_processed['Age'].median(), inplace=True)
    
    # Handle missing values in Embarked
    df_processed['Embarked'].fillna('S', inplace=True)
    
    # Handle missing values in Fare
    df_processed['Fare'].fillna(df_processed['Fare'].median(), inplace=True)
    
    # Create new features
    df_processed['FamilySize'] = df_processed['SibSp'] + df_processed['Parch'] + 1
    df_processed['IsAlone'] = (df_processed['FamilySize'] == 1).astype(int)
    
    # Extract title from name
    df_processed['Title'] = df_processed['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    df_processed['Title'] = df_processed['Title'].replace(['Lady', 'Countess','Capt', 'Col',
        'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df_processed['Title'] = df_processed['Title'].replace('Mlle', 'Miss')
    df_processed['Title'] = df_processed['Title'].replace('Ms', 'Miss')
    df_processed['Title'] = df_processed['Title'].replace('Mme', 'Mrs')
    
    # Fill missing titles
    df_processed['Title'].fillna('Mr', inplace=True)
    
    # Create age groups
    df_processed['AgeGroup'] = pd.cut(df_processed['Age'], bins=[0, 12, 18, 35, 60, 100], 
                                     labels=['Child', 'Teen', 'Young Adult', 'Adult', 'Senior'])
    
    # Create fare groups
    df_processed['FareGroup'] = pd.qcut(df_processed['Fare'], q=4, labels=['Low', 'Medium', 'High', 'Very High'])
    
    print("Preprocessing completed.")
    return df_processed


def apply_transformers(df: pd.DataFrame, transformers_dir: str) -> pd.DataFrame:
    """
    Apply saved transformers to preprocessed data.
    """
    print(f"Loading transformers from {transformers_dir}...")
    
    # Load transformers
    with open(os.path.join(transformers_dir, 'num_cat_transformer.pkl'), 'rb') as f:
        num_cat_transformer = pickle.load(f)
    
    with open(os.path.join(transformers_dir, 'bins_transformer.pkl'), 'rb') as f:
        bins_transformer = pickle.load(f)
    
    print("Applying transformations...")
    
    # Apply transformations in the same order as training
    df_transformed = num_cat_transformer.transform(df)
    df_final = bins_transformer.transform(df_transformed)
    
    print("Transformations applied successfully.")
    return df_final


def make_predictions(model, features: pd.DataFrame) -> pd.Series:
    """Make predictions using the trained model."""
    print("Making predictions...")
    predictions = model.predict(features)
    print(f"Generated {len(predictions)} predictions.")
    return predictions


def save_predictions(predictions: pd.Series, passenger_ids: pd.Series, output_path: str):
    """Save predictions to CSV file."""
    print(f"Saving predictions to {output_path}...")
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'PassengerId': passenger_ids,
        'Survived': predictions
    })
    
    # Save to CSV
    results_df.to_csv(output_path, index=False)
    print(f"Predictions saved successfully!")
    print(f"Results summary:")
    print(f"  Total predictions: {len(predictions)}")
    print(f"  Predicted survivors: {predictions.sum()}")
    print(f"  Predicted deaths: {len(predictions) - predictions.sum()}")


def main():
    """Main inference pipeline."""
    args = build_parser().parse_args()
    
    print("=" * 50)
    print("TITANIC SURVIVAL PREDICTION - INFERENCE")
    print("=" * 50)
    
    # Load inference data
    print(f"Loading inference data from {args.input_data}...")
    df_raw = pd.read_csv(args.input_data)
    print(f"Loaded {len(df_raw)} samples for inference.")
    
    # Store PassengerId for output
    passenger_ids = df_raw['PassengerId'].copy()
    
    # Remove PassengerId from features (not used in model)
    df_features = df_raw.drop('PassengerId', axis=1)
    
    # Apply preprocessing
    df_preprocessed = preprocess_inference_data(df_features)
    
    # Apply transformers
    df_transformed = apply_transformers(df_preprocessed, args.transformers_dir)
    
    # Load model
    model = load_model(args.model)
    
    # Make predictions
    predictions = make_predictions(model, df_transformed)
    
    # Save results
    save_predictions(predictions, passenger_ids, args.output)
    
    print("=" * 50)
    print("INFERENCE COMPLETED SUCCESSFULLY!")
    print("=" * 50)


if __name__ == "__main__":
    main()
