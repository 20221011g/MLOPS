from kedro.pipeline import Pipeline, node
from .nodes import predict_linear_model, predict_decision_tree_model

def create_predict_pipeline(**kwargs) -> Pipeline:
    return pipeline(
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
                func= model_predict,
                inputs=["test_model","X_test_data","y_test_data","housing_daily_data_engineered","parameters","best_columns"],
                outputs= "daily_prediction",
                name="predict",
            ),
        ]
    )
