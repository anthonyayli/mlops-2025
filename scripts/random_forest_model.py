import argparse
import pandas as pd
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, MinMaxScaler, KBinsDiscretizer
from sklearn.pipeline import Pipeline
from featurize import get_cat_tranformation, get_bins
from base_model_interface import ModelInterface

class RandomForestModel(ModelInterface):
    def load_data(self, features_path, labels_path):
        X = pd.read_csv(features_path)
        y = pd.read_csv(labels_path)
        if isinstance(y, pd.DataFrame):
            y = y.iloc[:, 0]
        return X, y

    def train_model(self, X_train, y_train):
        num_cat_tranformation = get_cat_tranformation()
        bins = get_bins()
        
        pipeline = Pipeline([
            ('num_cat_transform', num_cat_tranformation),
            ('binning', bins),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
        ])
        
        pipeline.fit(X_train, y_train)
        return pipeline

    def save_model(self, model, model_path):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

def build_parser():
    p = argparse.ArgumentParser(description="Train RandomForest model on Titanic dataset")
    p.add_argument("--train-features", required=True, help="Path to training features CSV")
    p.add_argument("--train-labels", required=True, help="Path to training labels CSV")
    p.add_argument("--model-output", default="models/random_forest_model.pkl", help="Path to save trained model")
    return p

def main():
    args = build_parser().parse_args()
    rf_model = RandomForestModel()
    
    # Load data
    X_train, y_train = rf_model.load_data(args.train_features, args.train_labels)
    
    # Train model
    model = rf_model.train_model(X_train, y_train)
    rf_model.save_model(model, args.model_output)
    
    print("RandomForest model trained and saved successfully!")
    print(f"Model saved to: {args.model_output}")
    print("Use evaluate.py to evaluate the model performance.")

if __name__ == "__main__":
    main()
