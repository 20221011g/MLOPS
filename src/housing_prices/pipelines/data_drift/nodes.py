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
import nannyml as nml

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

def house_price_drift(data_reference: pd.DataFrame, data_analysis: pd.DataFrame):
    # Define the threshold for the test
    constant_threshold = nml.thresholds.ConstantThreshold(lower=0.3, upper=0.7)
    constant_threshold.thresholds(data_reference)

    # Initialize the object that will perform the Univariate Drift calculations
    univariate_calculator = nml.UnivariateDriftCalculator(
        column_names=["house_price"],
        treat_as_categorical=['house_price'],
        chunk_size=50,
        categorical_methods=['jensen_shannon'],
        thresholds={"jensen_shannon": constant_threshold}
    )

    # Fit to reference data and calculate drift
    univariate_calculator.fit(data_reference)
    results = univariate_calculator.calculate(data_analysis).to_df()

    # Generate a report
    data_drift_report = evidently.dashboard.Report(metrics=[
        evidently.model_profile.DataDriftPreset(cat_stattest='ks', stattest_threshold=0.05)
    ])

    data_drift_report.run(
        current_data=data_analysis[["house_price"]],
        reference_data=data_reference[["house_price"]],
        column_mapping=None
    )
    data_drift_report.save_html("data/reporting/house_price_drift_report.html")

    return results
