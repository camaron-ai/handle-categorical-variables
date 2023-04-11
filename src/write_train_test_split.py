import pandas as pd
import numpy as np
import cfg
from sklearn.model_selection import train_test_split
import logging
from src import util

logger = logging.getLogger(__name__)


def main(env: cfg.EnviromentManager, output_path: str):
    train_data = pd.read_parquet(env.clean_train_path, columns=[cfg.TARGET_NAME])
    fold_indices = pd.DataFrame(index=train_data.index)
    indices = np.arange(len(train_data))
    train_idx, valid_idx = train_test_split(indices, stratify=train_data[cfg.TARGET_NAME], random_state=cfg.MAIN_SEED)
    fold_indices['is_test'] = -1
    fold_indices.iloc[train_idx, 0] = 0
    fold_indices.iloc[valid_idx, 0] = 1

    # save it to output path
    logger.info(f'output path: {output_path}')
    util.create_dir_from_path(output_path)
    print(fold_indices.head().to_markdown())
    fold_indices.to_csv(output_path, index=False)


if __name__ == '__main__':
    util.setup_logging()
    env = cfg.EnviromentManager.from_enviroment()
    main(env, env.cv_path)