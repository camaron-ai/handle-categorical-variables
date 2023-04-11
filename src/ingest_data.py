import pandas as pd
import numpy as np
import cfg
from sklearn.model_selection import train_test_split
import logging
from src import util
from sklearn.preprocessing import minmax_scale

logger = logging.getLogger(__name__)


def main(env: cfg.EnviromentManager):
    data = pd.read_csv(env.raw_train_path)
    for feat_name in cfg.VARIABLES_TO_MIN_MAX_SCALE:
        logger.info(f'min max scale {feat_name}')
        data[feat_name] = minmax_scale(data[feat_name]).astype(np.float32)

    util.create_dir_from_path(env.clean_train_path)
    logger.info(f'output path: {env.clean_train_path}')
    data.to_parquet(env.clean_train_path, index=False)

if __name__ == '__main__':
    util.setup_logging()
    env = cfg.EnviromentManager.from_enviroment()
    main(env)