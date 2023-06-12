from typing import Dict
from pandas import DataFrame
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
import numpy as np

def predict_linear_model(model: LinearRegression, X_test: DataFrame, parameters: Dict[str, Any], best_cols):
    if  parameters["with_feature_selection"] == False:
        X_test_temp = X_test.copy()
    else:
        X_test_temp = X_test[best_cols].copy()

    predictions = model.predict(X_test_temp)
    return predictions

def predict_decision_tree_model(model: DecisionTreeRegressor, X_test: DataFrame, parameters: Dict[str, Any], best_cols):
    if  parameters["with_feature_selection"] == False:
        X_test_temp = X_test.copy()
    else:
        X_test_temp = X_test[best_cols].copy()

    predictions = model.predict(X_test_temp)
    return predictions
