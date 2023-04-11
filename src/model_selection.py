import pandas as pd
import numpy as np
from typing import Tuple

TrainTestIndices = Tuple[np.ndarray, np.ndarray]

def get_train_test_indices(
    data: pd.DataFrame,
    cv_split_path: str,
    ) -> TrainTestIndices:

    df_split = pd.read_csv(cv_split_path)

    assert len(data) == len(df_split)
    train_mask = df_split.iloc[:, 0] == 0
    test_mask = df_split.iloc[:, 0] == 1
    indices = np.arange(len(data))
    assert (test_mask & train_mask).sum() == 0, 'leakage'
    return indices[train_mask], indices[test_mask]