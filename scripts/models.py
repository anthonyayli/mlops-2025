import argparse
import pandas as pd
import pickle
import os
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, MinMaxScaler, KBinsDiscretizer
from sklearn.pipeline import Pipeline
from featurize import get_num_cat_tranformation, get_bins
def build_parser():
    p = argparse.ArgumentParser(description="Train model on Titanic dataset")
    p.add_argument("--train-features", required=True, help="Path to training features CSV")
    p.add_argument("--train-labels", required=True, help="Path to training labels CSV")
    p.add_argument("--model-output", default="models/model.pkl", help="Path to save trained model")
    return p


def load_data(features_path, labels_path):
    X = pd.read_csv(features_path)
    y = pd.read_csv(labels_path)
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]
    return X, y

def train_model(X_train, y_train):
    num_cat_tranformation = get_num_cat_tranformation()
    bins = get_bins()
    
    pipeline = Pipeline([
        ('num_cat_transform', num_cat_tranformation),
        ('binning', bins),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ])
    
    pipeline.fit(X_train, y_train)
    return pipeline

def save_model(model, model_path):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

def main():
    args = build_parser().parse_args()
    X_train, y_train = load_data(args.train_features, args.train_labels)
    model = train_model(X_train, y_train)
    save_model(model, args.model_output)

if __name__ == "__main__":
    main()