"""
This is a boilerplate pipeline
generated using Kedro 0.18.8
"""

import logging
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import shap 
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
import shap
import mlflow
import mlflow.sklearn
import logging
import mlflow

def linear_model_train(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.DataFrame, y_test: pd.DataFrame, parameters: Dict[str, Any], best_cols):
    
    if  parameters["with_feature_selection"] == False:
        X_train_temp = X_train.copy()
        X_test_temp = X_test.copy()
        best_cols = list(X_train.columns)
    else:
        X_train_temp = X_train[best_cols].copy()
        X_test_temp = X_test[best_cols].copy()


    # params as found in gridsearch on data exploration
    model = LinearRegression(copy_X= True, fit_intercept= True)
    model.fit(X_train_temp, y_train)
    mlflow.set_tag("mlflow.runName", parameters["run_name"])
    #mlflow.autolog(log_model_signatures=True, log_input_examples=True)
    mlflow.sklearn.log_model(model, "model")

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


def random_forest_model_train(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.DataFrame, y_test: pd.DataFrame, parameters: Dict[str, Any], best_cols):
    
    if  parameters["with_feature_selection"] == False:
        X_train_temp = X_train.copy()
        X_test_temp = X_test.copy()
        best_cols = list(X_train.columns)
    else:
        X_train_temp = X_train[best_cols].copy()
        X_test_temp = X_test[best_cols].copy()

    # params as found in gridsearch on data exploration
    model = RandomForestRegressor(max_depth=None, min_samples_split=2, n_estimators=300, random_state=42)
    model.fit(X_train_temp, y_train)

    mlflow.set_tag("mlflow.runName", parameters["run_name"])
    mlflow.sklearn.log_model(model, "model")

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


def gradient_boosting_model_train(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.DataFrame, y_test: pd.DataFrame, parameters: Dict[str, Any], best_cols):
    
    if  parameters["with_feature_selection"] == False:
        X_train_temp = X_train.copy()
        X_test_temp = X_test.copy()
        best_cols = list(X_train.columns)
    else:
        X_train_temp = X_train[best_cols].copy()
        X_test_temp = X_test[best_cols].copy()

    # params as found in gridsearch on data exploration
    model = GradientBoostingRegressor(learning_rate=0.1, max_depth=5, min_samples_split=2, n_estimators=100, random_state=42)
    model.fit(X_train_temp, y_train)

    mlflow.set_tag("mlflow.runName", parameters["run_name"])
    mlflow.sklearn.log_model(model, "model")

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
