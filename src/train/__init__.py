"""
module for training machine learning models

implemented models are:
- xgboost
- logistic regression
- MLP neuro network
"""


from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import sklearn
import pandas as pd
from typing import List, Dict, Any
import time
import logging

logger = logging.getLogger(__name__)
SklearnClassifier = sklearn.base.ClassifierMixin

MODEL_DISPATCHER = {'logreg': LogisticRegression, 'xgb': XGBClassifier}



def load_model_from_config(config: Dict[str, Any]):
    model_instance_str = config['model']
    parameters = config['model_params']

    model_instance = MODEL_DISPATCHER[model_instance_str]
    return model_instance(**parameters)



def train_model(
    model: SklearnClassifier,
    data: pd.DataFrame,
    feature_names: List[str],
    target_name: str) -> SklearnClassifier:
    start_time = time.time()
    model.fit(data.loc[:, feature_names], data.loc[:, target_name])
    elapsed_time = time.time() - start_time
    logger.info(f'elapsed time: {elapsed_time:.4f}s')
    return model


def predict( model: SklearnClassifier,
    test_data: pd.DataFrame,
    feature_names: List[str]):
    return model.predict_proba(test_data.loc[:, feature_names])[:, 1]