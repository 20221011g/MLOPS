import pandas as pd
import numpy as np
from typing import Tuple, Dict
from sklearn.impute import SimpleImputer


def clean_data(
        data: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Does some data cleaning.
    Args:
        data: Data containing features and target.
    Returns:
        data: Cleaned data
    """
    # remove the column id
    data = data.drop('Id', axis=1)

    if 'SalePrice' in data.columns:
        # Remove outliers based on a specific condition for 'SalePrice'
        data = remove_outliers(data, 'SalePrice', 700000)
        # Rename 'SalePrice' column to 'target'
        data = data.rename(columns={'SalePrice': 'target'})

    # Remove outliers for other columns
    data = remove_outliers(data, 'TotalBsmtSF', 5000)
    data = remove_outliers(data, 'LotArea', 100000)

    # Create a copy of the DataFrame for further cleaning
    df_transformed = data.copy()
    describe_to_dict = pd.DataFrame(df_transformed.describe().to_dict())

    # Drop rows with missing values in columns other than 'target'
    data.dropna(subset=df_transformed.columns[df_transformed.columns != 'target'], inplace=True)

    if 'target' in df_transformed.columns:
        # Impute missing values in 'target' column using the mean strategy
        imputer = SimpleImputer(strategy='mean')
        target = df_transformed['target'].values.reshape(-1, 1)
        imputer.fit(target)
        df_transformed['target'] = imputer.transform(target)

    # Perform one-hot encoding on categorical columns
    cat_cols = ['MSSubClass', 'MSZoning', 'LotConfig', 'BldgType', 'Exterior1st']
    df_transformed = pd.get_dummies(df_transformed, columns=cat_cols)

    # Calculate descriptive statistics of the transformed DataFrame
    describe_to_dict_verified = pd.DataFrame(df_transformed.describe().to_dict())

    return df_transformed, describe_to_dict, describe_to_dict_verified


def feature_engineer(data: pd.DataFrame) -> pd.DataFrame:
    # delete null columns and rows with null values
    data = data.dropna(axis=1, how='all')
    data = data.dropna(axis=0, how='any')

    # Ensure all other columns are of a numeric type, else convert them
    for col in data.columns:
        if data[col].dtype not in ['int64', 'float64']:
            data[col] = pd.to_numeric(data[col], errors='coerce')

    return data

def remove_outliers(data, col, val):
    for index, value in data[col].items():
        if value > val:
            data.drop(index, inplace=True)
    print(len(data))
    return data
