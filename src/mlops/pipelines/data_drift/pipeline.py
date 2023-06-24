
"""
This is a boilerplate pipeline
generated using Kedro 0.18.8
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import  data_drift
from mlops.pipelines.data_preprocessing.nodes import clean_data, feature_engineer

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=clean_data,
                inputs="housing_daily_data",
                outputs=["housing_daily_data_cleaned","raw_describe_daily","cleaned_describe_daily"],
                name="clean_daily",
            ),

            node(
                func= feature_engineer,
                inputs="housing_daily_data_cleaned",
                outputs= "housing_daily_data_engineered",
                name="engineering_daily",
            ),

            node(
                func= data_drift,
                inputs=["housing_raw_data", "housing_daily_data"],
                outputs= "house_price_drift_report",
                name="data_drift",
            ),
        ]
    )
