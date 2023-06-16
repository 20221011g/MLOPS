
"""
This is a boilerplate pipeline
generated using Kedro 0.18.8
"""
import pandas as pd
import numpy as np
from typing import Tuple, Dict

from kedro.extras.datasets.pandas import CSVDataSet
from kedro.io import DataCatalog
from kedro.runner import SequentialRunner
#from kedro.catalog import DataCatalog
from kedro.pipeline import Pipeline, node, pipeline
from sklearn.impute import SimpleImputer
from .nodes import clean_data, feature_engineer

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=clean_data,
                inputs="housing_raw_data",
                outputs=["cleaned_data","raw_describe","cleaned_describe"],
                name="clean",
            ),

            node(
                func= feature_engineer,
                inputs="cleaned_data",
                outputs= "housing_data_engineered",
                name="engineering",
            ),

        ]
    )
