from kedro.pipeline import Pipeline, node
from .nodes import predict_linear_model, predict_decision_tree_model
from src.mlops.pipelines.data_preprocessing.nodes import clean_data, feature_engineer
from src.mlops.pipelines.data_split.nodes import split_data

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=clean_data,
                inputs="housing_daily_data",
                outputs=["housing_daily_data_cleaned","raw_describe_daily","cleaned_describe_daily"],
                name="clean",
            ),

            node(
                func= feature_engineer,
                inputs="housing_daily_data_cleaned",
                outputs= "housing_daily_data_engineered",
                name="engineering",
            ),

            node(
                func=split_data,
                inputs="housing_daily_data_engineered",
                outputs=["X_train_pred","X_test_pred","y_train_pred","y_test_pred"],
                name="split",
            ),
            
            node(
                func= predict_linear_model,
                inputs=["linear_model", "X_test_pred", "parameters", "params:best_columns"],
                outputs= "daily_prediction",
                name="predict",
            ),

        ]
    )
