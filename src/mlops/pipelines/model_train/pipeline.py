from kedro.pipeline import Pipeline, node
from .nodes import linear_model_train
from .nodes import decision_tree_model_train

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
                func=decision_tree_model_train,
                inputs=["X_train", "X_test", "y_train", "y_test", "parameters", "params:best_cols"],
                outputs=["trained_decision_tree_model", "decision_tree_model_plot"],
                name="train_decision_tree_model",
            ),
        ]
    )
