import pandas as pd
import numpy as np
import cfg
import logging
from typing import Dict
from src.model_selection import get_train_test_indices
import os
from src import util
from src import train
from src import tracking
import copy
from sklearn import metrics
import typer
import pprint

logger = logging.getLogger(__name__)


def compute_metrics(y: np.ndarray, yhat: np.ndarray, prefix: str = '') -> Dict[str, float]:


    scores = {
        f'{prefix}logloss': metrics.log_loss(y, yhat),
        f'{prefix}auc': metrics.roc_auc_score(y, yhat)
    }
    return scores


def run_experiment_from_config(dataset_name: str, confpath: str):

    env = cfg.EnviromentManager.from_enviroment()
    util.seed_everything(cfg.MAIN_SEED)

    logger.info('load target data..')
    target_data = pd.read_csv(env.raw_train_path, usecols=[cfg.TARGET_NAME])

    model_name = util.get_filename(confpath)
    experiment_name = f'{dataset_name}-{model_name}'
    logger.info(f'running experiment {experiment_name}')

    config = util.load_config(confpath)
    train_idx, valid_idx = get_train_test_indices(target_data, env.cv_path)
    
    logger.info('setting up train data..')
    input_ft_dir = env.format_features_dir(dataset_name)
    train_target = target_data.iloc[train_idx].reset_index(drop=True)
    train_ft_path = os.path.join(input_ft_dir, 'train.parquet')
    train_features = pd.read_parquet(train_ft_path)
    assert len(train_features) == len(train_target)
    train_data = pd.concat((train_target, train_features), axis=1)

    # no nans
    assert train_data.isna().sum().sum() == 0

    feature_names = list(train_features.columns)

    logger.info(f'# of input features: {len(feature_names)}')

    print(train_data.head())

    logger.info('training model')
    model = train.load_model_from_config(config)
    model = train.train_model(model, train_data, feature_names, target_name=cfg.TARGET_NAME)

    logger.info('setting up valid data..')
    valid_target = target_data.iloc[valid_idx].reset_index(drop=True)
    valid_ft_path = os.path.join(input_ft_dir, 'valid.parquet')
    valid_features = pd.read_parquet(valid_ft_path)
    assert len(valid_features) == len(valid_target)
    valid_data = pd.concat((valid_target, valid_features), axis=1)

    # no nans
    assert valid_data.isna().sum().sum() == 0

    # predict
    train_yhat = train.predict(model, train_data, feature_names)
    valid_yhat = train.predict(model, valid_data, feature_names)

    # compute scores
    train_scores = compute_metrics(train_data[cfg.TARGET_NAME], train_yhat, 'train_')
    valid_scores = compute_metrics(valid_data[cfg.TARGET_NAME], valid_yhat, 'valid_')

    scores = dict(**train_scores, **valid_scores)
    mlflow_paramters = copy.deepcopy(config['model_params'])
    mlflow_paramters['dataset_name'] = dataset_name
    mlflow_paramters['model'] = config['model']
    mlflow_paramters['n_input_features'] = len(feature_names)


    pprint.pprint(scores)
    tracking.track_experiment(
        experiment_name,
        cfg.MLFLOW_DEFAULT_EXPERIMENT,
        mlflow_paramters,
        scores,
        )


if __name__ == '__main__':
    util.setup_logging()
    typer.run(run_experiment_from_config)
