import argparse
from dataclasses import dataclass
import pandas as pd
import numpy as np
import warnings

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.feature_selection import SelectKBest,chi2
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import KBinsDiscretizer

warnings.filterwarnings("ignore")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Feature engineering for Titanic dataset")
    p.add_argument("--input-frame", required=True, help="Path to dataframe CSV file")
    p.add_argument("--output-train-features", required=True, help="Path to save training features")
    p.add_argument("--output-train-labels", required=True, help="Path to save training labels")
    p.add_argument("--output-test-features", required=True, help="Path to save test features")
    p.add_argument("--output-test-labels", required=True, help="Path to save test labels")
    return p
def engineering_data(df):
    data = pd.read_csv(df)
    data['Survived'] = data['Survived'].astype('int64')
    data = data.drop("PassengerId", axis=1)
    X = data.drop("Survived", axis=1)
    Y = data["Survived"]
    X_train, X_test, y_train, y_test=train_test_split(X,Y,test_size=0.2)
    return X_train, X_test, y_train, y_test



def get_cat_tranformation():
    num_cat_tranformation=ColumnTransformer([
                                    ('scaling',MinMaxScaler(),[0,2]),
                                    ('onehotencolding1',OneHotEncoder(),[1,3]),
                                    ('ordinal',OrdinalEncoder(),[4]),
                                    ('onehotencolding2',OneHotEncoder(),[5,6])
                                    ],remainder='passthrough')
    return num_cat_tranformation
def get_bins():
    bins=ColumnTransformer([
                        ('Kbins',KBinsDiscretizer(n_bins=15,encode='ordinal',strategy='quantile'),[0,2]),
                        ],remainder='passthrough')
    return bins


def main():
    args = build_parser().parse_args()
    
    X_train, X_test, y_train, y_test = engineering_data(args.input_frame)
    
    X_train.to_csv(args.output_train_features, index=False)
    y_train.to_csv(args.output_train_labels, index=False)
    X_test.to_csv(args.output_test_features, index=False)
    y_test.to_csv(args.output_test_labels, index=False)
    
 


if __name__ == "__main__":
    main()

