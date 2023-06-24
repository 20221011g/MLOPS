from typing import Dict, Any
import pandas as pd
from nannyml import UnivariateDriftCalculator, thresholds
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from html_dataset import HTMLDataSet


def data_drift(data_reference: pd.DataFrame, data_analysis: pd.DataFrame) -> Dict[str, Any]:
    # Specify the columns to include in the drift analysis
    columns = ['MSSubClass', 'LotArea', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'TotalBsmtSF']
    data_reference = data_reference[columns].fillna(0)
    data_reference = data_reference[columns].astype(int)

    data_analysis = data_analysis[columns].fillna(0)
    data_analysis = data_analysis[columns].astype(int)

    # Define the threshold for the data_analysis
    constant_threshold = thresholds.ConstantThreshold(lower=0.3, upper=0.7)
    constant_threshold.thresholds(data_reference)

    # Initialize the univariate drift calculator
    univariate_calculator = UnivariateDriftCalculator(
        column_names=columns,
        treat_as_categorical=columns,
        chunk_size=50,
        categorical_methods=['jensen_shannon'],
        thresholds={"jensen_shannon": constant_threshold}
    )

    # Fit the univariate calculator on the data_referenceing data
    univariate_calculator.fit(data_reference)

    # Check for univariate data drift in the data_analysis data
    univariate_drift_report = univariate_calculator.calculate(data_analysis).to_df()

    # Generate a data drift report using Evidently AI
    data_drift_report = Report(metrics=[DataDriftPreset(cat_stattest='ks', stattest_threshold=0.05)])
    data_drift_report.run(current_data=data_analysis, reference_data=data_reference, column_mapping=None)

    # Return the univariate drift report
    return univariate_drift_report