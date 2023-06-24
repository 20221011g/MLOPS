from kedro.pipeline import Pipeline, node
from .nodes import predict_linear_model, predict_decision_tree_model
from src.mlops.pipelines.data_preprocessing.nodes import clean_data, feature_engineer
from src.mlops.pipelines.data_split.nodes import split_data

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func= predict_linear_model,
                inputs=["trained_linear_model", "housing_daily_data_engineered", "parameters", "params:best_cols"],
                outputs= "daily_prediction",
                name="predict",
            ),

        ]
    )
