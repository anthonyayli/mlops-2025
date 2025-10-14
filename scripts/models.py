import argparse
import pandas as pd
import pickle
import os
from sklearn.linear_model import LogisticRegression
# Transformations are now handled in featurize.py
from base_model_interface import ModelInterface

class TitanicModel(ModelInterface):
    def load_data(self, features_path, labels_path):
        X = pd.read_csv(features_path)
        y = pd.read_csv(labels_path)
        if isinstance(y, pd.DataFrame):
            y = y.iloc[:, 0]
        # Convert to numpy arrays since features are already transformed
        X = X.values
        y = y.values if hasattr(y, 'values') else y
        return X, y

    def train_model(self, X_train, y_train):
        # Features are already transformed in featurize.py
        # Just train the classifier directly
        classifier = LogisticRegression(max_iter=1000, random_state=42)
        classifier.fit(X_train, y_train)
        return classifier

    def save_model(self, model, model_path):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

def build_parser():
    p = argparse.ArgumentParser(description="Train model on Titanic dataset")
    p.add_argument("--train-features", required=True, help="Path to training features CSV")
    p.add_argument("--train-labels", required=True, help="Path to training labels CSV")
    p.add_argument("--model-output", default="models/model.pkl", help="Path to save trained model")
    return p

def main():
    args = build_parser().parse_args()
    titanic_model = TitanicModel()
    X_train, y_train = titanic_model.load_data(args.train_features, args.train_labels)
    model = titanic_model.train_model(X_train, y_train)
    titanic_model.save_model(model, args.model_output)

if __name__ == "__main__":
    main()