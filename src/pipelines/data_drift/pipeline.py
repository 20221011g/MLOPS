
"""
This is a boilerplate pipeline
generated using Kedro 0.18.8
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import  data_drift
from project_template.pipelines.data_preprocessing.nodes import clean_data, feature_engineer

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=clean_data,
                inputs="",
                outputs=[],
                name="",
            ),

            node(
                func= feature_engineer,
                inputs="",
                outputs= "",
                name="",
            ),

            node(
                func= data_drift,
                inputs=[],
                outputs= "",
                name="",
            ),
        ]
    )
