import argparse
import pandas as pd
import warnings
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import KBinsDiscretizer
from featurize_interface import FeaturizerInterface

warnings.filterwarnings("ignore")


class TitanicFeaturizer(FeaturizerInterface):
    
    def engineering_data(self, df):
        data = pd.read_csv(df)
        train_data = data.dropna(subset=['Survived'])
        train_data['Survived'] = train_data['Survived'].astype('int64')
        train_data = train_data.drop("PassengerId", axis=1)
        X = train_data.drop("Survived", axis=1)
        Y = train_data["Survived"]
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
        
        # Apply transformations
        num_cat_transformation = self.get_cat_tranformation()
        bins = self.get_bins()
        
        # Transform training data
        X_train_transformed = num_cat_transformation.fit_transform(X_train)
        X_train_final = bins.fit_transform(X_train_transformed)
        
        # Transform test data using fitted transformers
        X_test_transformed = num_cat_transformation.transform(X_test)
        X_test_final = bins.transform(X_test_transformed)
        
        # Convert back to DataFrames for CSV saving
        X_train_df = pd.DataFrame(X_train_final)
        X_test_df = pd.DataFrame(X_test_final)
        
        return X_train_df, X_test_df, y_train, y_test
    
    def get_cat_tranformation(self):
        num_cat_tranformation = ColumnTransformer([
            ('scaling', MinMaxScaler(), [0, 2]),
            ('onehotencolding1', OneHotEncoder(), [1, 3]),
            ('ordinal', OrdinalEncoder(), [4]),
            ('onehotencolding2', OneHotEncoder(), [5, 6])
        ], remainder='passthrough')
        return num_cat_tranformation
    
    def get_bins(self):
        bins = ColumnTransformer([
            ('Kbins', KBinsDiscretizer(n_bins=15, encode='ordinal', strategy='quantile'), [0, 2]),
        ], remainder='passthrough')
        return bins


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Feature engineering for Titanic dataset")
    p.add_argument("--input-frame", required=True, help="Path to dataframe CSV file")
    p.add_argument("--output-train-features", required=True, help="Path to save training features")
    p.add_argument("--output-train-labels", required=True, help="Path to save training labels")
    p.add_argument("--output-test-features", required=True, help="Path to save test features")
    p.add_argument("--output-test-labels", required=True, help="Path to save test labels")
    return p


def get_cat_tranformation():
    featurizer = TitanicFeaturizer()
    return featurizer.get_cat_tranformation()


def get_bins():
    featurizer = TitanicFeaturizer()
    return featurizer.get_bins()


def main():
    args = build_parser().parse_args()
    
    featurizer = TitanicFeaturizer()
    X_train, X_test, y_train, y_test = featurizer.engineering_data(args.input_frame)
    
    X_train.to_csv(args.output_train_features, index=False)
    y_train.to_csv(args.output_train_labels, index=False)
    X_test.to_csv(args.output_test_features, index=False)
    y_test.to_csv(args.output_test_labels, index=False)


if __name__ == "__main__":
    main()