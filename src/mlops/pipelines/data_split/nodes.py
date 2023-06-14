"""
This is a boilerplate pipeline
generated using Kedro 0.18.8
"""

import logging
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd



from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder , LabelEncoder
import shap 
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
import sklearn
import mlflow


def split_data(
    data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Splits data into features and target training and test sets.

    Args:
        data: Data containing features and target.
        parameters: Parameters defined in parameters.yml.
    Returns:
        Split data.
    """

    assert [col for col in data.columns if data[col].isnull().any()] == []
    y = data["SalePrice"]
    X = data.drop(columns=["SalePrice"], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)
    X_train = data_train.drop(columns=["SalePrice"])
    X_test = data_test.drop(columns=["SalePrice"])
    y_train = data_train["SalePrice"]
    y_test = data_test["SalePrice"]
    return X_train, X_test, y_train, y_test
