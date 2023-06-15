import pandas as pd
import numpy as np
from typing import Tuple, Dict
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import VarianceThreshold


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
    clean_data = data.copy()
    describe_to_dict = clean_data.describe().to_dict()

    # Drop rows with missing values in columns other than 'SalePrice'
    data.dropna(subset=clean_data.columns[clean_data.columns != 'target'], inplace=True)

    # Impute missing values in 'target' column using the mean strategy
    imputer = SimpleImputer(strategy='mean')
    sale_price = clean_data['target'].values.reshape(-1, 1)
    imputer.fit(sale_price)
    clean_data['target'] = imputer.transform(sale_price)

    # Perform one-hot encoding on categorical columns
    cat_cols = ['MSSubClass', 'MSZoning', 'LotConfig', 'BldgType']
    clean_data = pd.get_dummies(clean_data, columns=cat_cols)

    # Calculate descriptive statistics of the transformed DataFrame
    describe_to_dict_verified = clean_data.describe().to_dict()
    print(len(clean_data))
    return cleaned_data, describe_to_dict, describe_to_dict_verified


def feature_engineer(
        cleaned_data: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict]:
    """Does some data cleaning.
    Args:
        data: Data containing features and target.
    Returns:
        data: Data after feature
    """
    # Turn 'BsmtFinSF2' into a binary feature
    cleaned_data['BsmtFinSF2'] = (cleaned_data['BsmtFinSF2'] > 0).astype(int)

    # Drop rows with NaN values
    cleaned_data.dropna(inplace=True)

    # Drop target from dataset
    cleaned_data_no_target = cleaned_data.drop(labels=['target'], axis=1)

    # create features and target
    X = cleaned_data_no_target
    y = cleaned_data['target']

    # convert to categorical data by converting data to integers
    X = X.astype(int)

    # Compute the mutual information between each feature and the target variable
    mi_scores = mutual_info_regression(X, y)

    # Create a dictionary to store feature scores
    feature_scores = dict(zip(X.columns, mi_scores))

    # Filter features with scores greater than 0
    best_info_columns = [feature for feature, score in feature_scores.items() if score > 0]

    # Make dataframe with the best columns
    X = X[best_info_columns]

    # using sklearn variancethreshold to find nearly-constant features
    sel = VarianceThreshold(threshold=0.01)
    sel.fit(X)  # fit finds the features with very small variance

    # Get the indices of included features
    best_columns = X.columns[sel.get_support()]

    # Transform X to include only the selected features
    X = sel.transform(X)

    # Convert X back to a DataFrame with column names
    data_engineered = pd.DataFrame(X, columns=best_columns)

    # Calculate descriptive statistics of the transformed DataFrame
    describe_to_dict_verified = data_engineered.describe().to_dict()
    print(len(data_engineered))
    return data_engineered, describe_to_dict_verified




