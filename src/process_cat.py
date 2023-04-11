import cfg
import logging
import pandas as pd
import os
import cfg
from src.model_selection import get_train_test_indices
from src import util
import typer
from sklearn.preprocessing import OneHotEncoder
import numpy as np

logger = logging.getLogger(__name__)

STRATEGY_DISPATCHER = {}

def label_encoding(train_data: pd.DataFrame, valid_data: pd.DataFrame):
    """
    categorical variables are already encoded to integers, so we dont have to do anything
    """
    return train_data, valid_data


def one_hot_encoding(train_data: pd.DataFrame, valid_data: pd.DataFrame):
    """
    encode categorical variables using one hot encoding strategy
    """

    ohe_tmf = OneHotEncoder(sparse=False, dtype=np.float32)    
    ohe_tmf.fit(train_data[cfg.CATEGORICAL_VARIABLES])

    ohe_columns = [
        f'{cat_name}_{cat_value}'
        for cat_name, categories_ in zip(cfg.CATEGORICAL_VARIABLES, ohe_tmf.categories_)
        for cat_value in categories_
    ]

    # process train data
    processed_train_data = train_data.drop(cfg.CATEGORICAL_VARIABLES, axis=1)
    _np_ohe_train = ohe_tmf.transform(train_data[cfg.CATEGORICAL_VARIABLES])    
    ohe_train = pd.DataFrame(_np_ohe_train, columns=ohe_columns)
    processed_train_data = pd.concat((processed_train_data, ohe_train), axis=1)

    # process valid data
    processed_valid_data = valid_data.drop(cfg.CATEGORICAL_VARIABLES, axis=1)
    _np_ohe_valid = ohe_tmf.transform(valid_data[cfg.CATEGORICAL_VARIABLES])    
    ohe_valid = pd.DataFrame(_np_ohe_valid, columns=ohe_columns)
    processed_valid_data = pd.concat((processed_valid_data, ohe_valid), axis=1)

    return processed_train_data, processed_valid_data



def freq_encoding(train_data: pd.DataFrame, valid_data: pd.DataFrame):
    """encode categorical variables by the frequencies of each category"""
    # becareful, this implementation does not handle unknown categorical values
    category_freq_mapper = {}

    for cat_name in cfg.CATEGORICAL_VARIABLES:
        # normalize so values are between 0 and 1!
        category_freq_mapper[cat_name] = train_data[cat_name].value_counts(normalize=True)

    processed_train_data = train_data.drop(cfg.CATEGORICAL_VARIABLES, axis=1)
    for cat_name in cfg.CATEGORICAL_VARIABLES:
        cat_freqs = category_freq_mapper[cat_name]
        processed_train_data[cat_name] = train_data[cat_name].map(cat_freqs).astype(np.float32)

    processed_valid_data = valid_data.drop(cfg.CATEGORICAL_VARIABLES, axis=1)
    for cat_name in cfg.CATEGORICAL_VARIABLES:
        cat_freqs = category_freq_mapper[cat_name]
        processed_valid_data[cat_name] = valid_data[cat_name].map(cat_freqs).astype(np.float32)
    
    return processed_train_data, processed_valid_data


STRATEGY_DISPATCHER['label_encoding'] = label_encoding
STRATEGY_DISPATCHER['one_hot_encoding'] = one_hot_encoding
STRATEGY_DISPATCHER['freq_encoding'] = freq_encoding



def main(strategy: str):
    logger.info(f'strategy: {strategy}')
    env = cfg.EnviromentManager.from_enviroment()
    logger.info('load data')
    data = pd.read_parquet(env.clean_train_path)
    data.head()
    logger.info('split into train and test set')
    train_idx, valid_idx = get_train_test_indices(data, env.cv_path)
    train_data = data.iloc[train_idx].reset_index(drop=True)
    valid_data = data.iloc[valid_idx].reset_index(drop=True)

    # get strategy function
    strategy_fn = STRATEGY_DISPATCHER[strategy]
    logger.info('process categrical variables')
    processed_train_data, processed_valid_data = strategy_fn(train_data, valid_data)

    # make sure id and target is not in the feature dataframe
    feature_names = util.get_allowed_columns(processed_train_data.columns)
    processed_train_data = processed_train_data[feature_names]
    processed_valid_data = processed_valid_data[feature_names]
    logger.info(f'# of features: {len(feature_names)}')

    output_dir = env.format_features_dir(strategy)
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f'output_dir: {output_dir}')


    assert processed_train_data.isna().sum().sum() == 0
    assert processed_valid_data.isna().sum().sum() == 0

    assert len(processed_train_data) == len(train_data)
    assert len(processed_valid_data) == len(valid_data)
    processed_train_data.to_parquet(os.path.join(output_dir, 'train.parquet'), index=False)
    processed_valid_data.to_parquet(os.path.join(output_dir, 'valid.parquet'), index=False)


if __name__ == '__main__':
    util.setup_logging()
    typer.run(main)