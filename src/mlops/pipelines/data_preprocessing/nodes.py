import pandas as pd
import numpy as np
from typing import Tuple, Dict
from sklearn.impute import SimpleImputer


def clean_data(
        data: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict, Dict]:
    """Does some data cleaning.
    Args:
        data: Data containing features and target.
    Returns:
        data: Cleaned data
    """
    # remove the column id
    data = data.drop('Id', axis=1)

    # Remove outliers based on a specific condition for 'SalePrice'
    data = remove_outliers(data, 'SalePrice', 700000)
    data = remove_outliers(data, 'TotalBsmtSF', 5000)
    data = remove_outliers(data, 'LotArea', 100000)

    # Rename 'SalePrice' column to 'target'
    data = data.rename(columns={'SalePrice': 'target'})

    # Create a copy of the DataFrame for further cleaning
    df_transformed = data.copy()
    describe_to_dict = df_transformed.describe().to_dict()

    # Drop rows with missing values in columns other than 'SalePrice'
    data.dropna(subset=df_transformed.columns[df_transformed.columns != 'target'], inplace=True)

    # Impute missing values in 'target' column using the mean strategy
    imputer = SimpleImputer(strategy='mean')
    sale_price = df_transformed['target'].values.reshape(-1, 1)
    imputer.fit(sale_price)
    df_transformed['target'] = imputer.transform(sale_price)

    # Perform one-hot encoding on categorical columns
    cat_cols = ['MSSubClass', 'MSZoning', 'LotConfig', 'BldgType']
    df_transformed = pd.get_dummies(df_transformed, columns=cat_cols)

    # Calculate descriptive statistics of the transformed DataFrame
    describe_to_dict_verified = df_transformed.describe().to_dict()
    print(len(df_transformed))
    return df_transformed, describe_to_dict, describe_to_dict_verified

def feature_engineer(data: pd.DataFrame) -> pd.DataFrame:
    return data

def remove_outliers(data, col, val):
    for index, value in data[col].items():
        if value > val:
            data.drop(index, inplace=True)
    print(len(data))
    return data






