"""
This is a boilerplate pipeline
generated using Kedro 0.18.8
"""

import logging
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
import shap 
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import sklearn
import mlflow

def linear_model_train(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.DataFrame, y_test: pd.DataFrame, parameters: Dict[str, Any], best_cols):
    
    if  parameters["with_feature_selection"] == False:
        X_train_temp = X_train.copy()
        X_test_temp = X_test.copy()
        best_cols = list(X_train.columns)
    else:
        X_train_temp = X_train[best_cols].copy()
        X_test_temp = X_test[best_cols].copy()

    mlflow.set_tag("mlflow.runName", parameters["run_name"])
    #mlflow.autolog(log_model_signatures=True, log_input_examples=True)

    model = LinearRegression()
    model.fit(X_train_temp, y_train)

    # Create object that can calculate shap values
    explainer = shap.Explainer(model, X_train_temp)

    # calculate shap values. This is what we will plot.
    shap_values = explainer.shap_values(X_test_temp)
    
    shap.summary_plot(shap_values, X_test_temp, show=False)

    preds = model.predict(X_test_temp)
    mse = mean_squared_error(y_test, preds)

    log = logging.getLogger(__name__)
    log.info(f"#Best columns: {len(best_cols)}")
    log.info("Model mean squared error on test set: %0.2f", mse)

    return model, plt



def decision_tree_model_train(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.DataFrame, y_test: pd.DataFrame, parameters: Dict[str, Any], best_cols):
    
    if  parameters["with_feature_selection"] == False:
        X_train_temp = X_train.copy()
        X_test_temp = X_test.copy()
        best_cols = list(X_train.columns)
    else:
        X_train_temp = X_train[best_cols].copy()
        X_test_temp = X_test[best_cols].copy()

    mlflow.set_tag("mlflow.runName", parameters["run_name"])

    model = DecisionTreeRegressor(max_depth=parameters.get("max_depth", None))
    model.fit(X_train_temp, y_train)

    # Create object that can calculate shap values
    explainer = shap.Explainer(model, X_train_temp)

    # calculate shap values. This is what we will plot.
    shap_values = explainer.shap_values(X_test_temp)
    
    shap.summary_plot(shap_values, X_test_temp, show=False)

    preds = model.predict(X_test_temp)
    mse = mean_squared_error(y_test, preds)

    log = logging.getLogger(__name__)
    log.info(f"#Best columns: {len(best_cols)}")
    log.info("Model mean squared error on test set: %0.2f", mse)

    return model, plt
