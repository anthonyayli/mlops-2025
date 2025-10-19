#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np
import warnings
from preprocessing_interface import PreprocessorInterface

warnings.filterwarnings("ignore")


class TitanicPreprocessor(PreprocessorInterface):
    
    def family_size(self, number):
        if number==1:
            return "Alone"
        elif number>1 and number <5:
            return "Small"
        else:
            return "Large"
    
    def preprocess_data(self, train_path, test_path):
        train = pd.read_csv(train_path)
        test = pd.read_csv(test_path)
        
        train.drop(columns=['Cabin'], inplace=True)
        test.drop(columns=['Cabin'], inplace=True)
        
        train['Embarked'].fillna('S', inplace=True)
        test['Fare'].fillna(test['Fare'].mean(), inplace=True)
        
        df = pd.concat([train, test], sort=True).reset_index(drop=True)
        
        df['Age'] = df.groupby(['Sex', 'Pclass'])['Age'].transform(lambda x: x.fillna(x.median()))
        df['Title']=df['Name'].str.split(", ",expand=True)[1].str.split(".",expand=True)[0]

        df['Age'] = df['Age'].astype('int64')
        df['Title'] = df['Title'].replace(['Lady', 'the Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
        df['Title'] = df['Title'].replace('Mlle', 'Miss')
        df['Title'] = df['Title'].replace('Ms', 'Miss')
        df['Title'] = df['Title'].replace('Mme', 'Mrs')
      
        df['Family_size']=df['SibSp'] + df['Parch'] + 1
        df.drop(columns=['Name','Parch','SibSp','Ticket'],inplace=True)
        df['Family_size']=df['Family_size'].apply(self.family_size)
        
        return df
    
    def prepare_train_test_data(self, df_processed):
        """Separate train and test data from processed dataframe"""
        # Get training data (rows with Survived values)
        train_data = df_processed.dropna(subset=['Survived']).copy()
        train_data['Survived'] = train_data['Survived'].astype('int64')
        train_data = train_data.drop("PassengerId", axis=1)
        
        # Get test data (rows without Survived values)
        test_data = df_processed[df_processed['Survived'].isna()].copy()
        test_data = test_data.drop("PassengerId", axis=1)
        test_data = test_data.drop("Survived", axis=1)  # Remove Survived column from test data
        
        return train_data, test_data


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Preprocess Titanic dataset")
    p.add_argument("--input-train", required=True, help="Path to training data CSV file")
    p.add_argument("--input-test", required=True, help="Path to test data CSV file")
    p.add_argument("--output", required=True, help="Path to save processed data")
    return p


def main():
    args = build_parser().parse_args()
    
    preprocessor = TitanicPreprocessor()
    df_processed = preprocessor.preprocess_data(args.input_train, args.input_test)
    
    df_processed.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()