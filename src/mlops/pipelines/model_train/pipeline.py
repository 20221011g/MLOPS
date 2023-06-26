from kedro.pipeline import Pipeline, node
from .nodes import linear_model_train
from .nodes import random_forest_model_train
from .nodes import gradient_boosting_model_train

def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=linear_model_train,
                inputs=["X_train", "X_test", "y_train", "y_test", "parameters", "params:best_cols"],
                outputs=["trained_linear_model", "linear_model_plot"],
                name="train_linear_model",
            ),
            node(
                func=random_forest_model_train,
                inputs=["X_train", "X_test", "y_train", "y_test", "parameters", "params:best_cols"],
                outputs=["trained_random_forest_model", "random_forest_model_plot"],
                name="train_random_forest_model",
            ),
            node(
                func=gradient_boosting_model_train,
                inputs=["X_train", "X_test", "y_train", "y_test", "parameters", "params:best_cols"],
                outputs=["trained_gradient_boosting_model", "gradient_boosting_model_plot"],
                name="train_gradient_boosting_model",
            ),
        ]
    )
