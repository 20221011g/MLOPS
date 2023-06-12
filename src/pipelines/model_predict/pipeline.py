from kedro.pipeline import Pipeline, node
from .nodes import predict_linear_model, predict_decision_tree_model

def create_predict_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=predict_linear_model,
                inputs=["linear_model", "X_test", "params:linear_regression", "params:best_cols"],
                outputs="linear_model_predictions",
                name="predict_linear_model",
            ),
            node(
                func=predict_decision_tree_model,
                inputs=["decision_tree_model", "X_test", "params:decision_tree", "params:best_cols"],
                outputs="decision_tree_model_predictions",
                name="predict_decision_tree_model",
            ),
        ]
    )
