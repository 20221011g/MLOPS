
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
from nodes import clean_data, feature_engineer
from kedro.pipeline import Pipeline, node, pipeline
from sklearn.impute import SimpleImputer
from nodes import clean_data, feature_engineer

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=clean_data,
                inputs="housing_raw_data",
                outputs=["housing_cleaned_data","raw_describe","cleaned_describe"],
                name="clean",
            ),

            #node(
             #   func= feature_engineer,
              #  inputs="housing_cleaned_data",
              #  outputs= "housing_data_engineered",
              #  name="engineering",
            #),

        ]
    )


# Create a data catalog with the required datasets
catalog = DataCatalog({"housing_raw_data": CSVDataSet(filepath="C:/Users/couto/PycharmProjects/MLOPS/data/01_raw/HousePricePrediction.csv")})

# Create an instance of the SequentialRunner
runner = SequentialRunner()

# Run the pipeline
runner.run(pipeline=create_pipeline(), catalog=catalog)
