
from typing import Dict
from kedro.pipeline import Pipeline, pipeline

from src.mlops.pipelines.data_preprocessing  import pipeline as preprocessing
from src.mlops.pipelines.data_split  import pipeline as split_data
from src.mlops.pipelines.model_train  import pipeline as train
#from src.mlops.pipelines.feature_selection  import pipeline as feature_selection
from src.mlops.pipelines.model_predict  import pipeline as predict
from src.mlops.pipelines.data_drift  import pipeline as data_drift


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """
    preprocessing_stage = preprocessing.create_pipeline()
    split_data_stage = split_data.create_pipeline()
    train_stage = train.create_pipeline()
    #feature_selection_stage = feature_selection.create_pipeline()
    predict_stage = predict.create_pipeline()
    drift_test_stage = data_drift.create_pipeline()


    return {
        "preprocessing": preprocessing_stage,
        "split_data": split_data_stage,
        "train": train_stage,
        #"feature_selection": feature_selection_stage,
        "predict": predict_stage,
        "drift_test" : drift_test_stage, 
        "__default__": preprocessing_stage + split_data_stage + train_stage
    }