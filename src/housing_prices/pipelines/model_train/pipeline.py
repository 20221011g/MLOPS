from kedro.pipeline import Pipeline, node
from .nodes import model_train

def create_train_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=model_train.linear_model_train,
                inputs=["X_train", "X_test", "y_train", "y_test", "params:linear_regression", "params:best_cols"],
                outputs=["linear_model", "linear_model_plot"],
                name="train_linear_model",
            ),
            node(
                func=model_train.decision_tree_model_train,
                inputs=["X_train", "X_test", "y_train", "y_test", "params:decision_tree", "params:best_cols"],
                outputs=["decision_tree_model", "decision_tree_model_plot"],
                name="train_decision_tree_model",
            ),
        ]
    )
