
from typing import Dict
from kedro.pipeline import Pipeline, pipeline

from pipelines import (
    data_preprocessing as preprocessing,
    data_split as split_data,
    model_train as train,
    feature_selection as best_features,
    model_predict as predict,
    data_drift as drift_test,

)

def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """
    preprocessing_stage = preprocessing.create_pipeline()
    split_data_stage = split_data.create_pipeline()
    train_stage = train.create_pipeline()
    feature_selection_stage = best_features.create_pipeline()
    predict_stage = predict.create_pipeline()
    drift_test_stage = drift_test.create_pipeline()


    return {
        "preprocessing": preprocessing_stage,
        "split_data": split_data_stage,
        "train": train_stage,
        "feature_selection": feature_selection_stage,
        "predict": predict_stage,
        "drift_test" : drift_test_stage, 
        "__default__": preprocessing_stage + split_data_stage + train_stage
    }