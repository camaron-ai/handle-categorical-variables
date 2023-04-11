import random
import numpy as np
import torch
import os
from typing import Union, Dict, Any, NamedTuple, List
from collections import namedtuple
import json
import yaml
from pathlib import Path
import logging
import coloredlogs
import sys

pathtype = Union[str, os.PathLike]
ExperimentInputData = namedtuple('ExperimentInputData', ['confpath', 'config', 'exp_name'])


def setup_logging(console_level=logging.INFO):
    """
    Setup logging configuration

        default_level (logging.LEVEL): default logging level
    Returns:
        None
    """
    console_format = ('%(asctime)s - %(name)s - %(levelname)-8s '
                      '[%(filename)s:%(lineno)d] %(message)s')
    root_logger = logging.getLogger()
    # Handlers for stdout/err logging
    output_handler = logging.StreamHandler(sys.stdout)
    output_handler.setFormatter(logging.Formatter(console_format))
    output_handler.setLevel(logging.DEBUG)
    root_logger.addHandler(output_handler)
    root_logger.setLevel(logging.DEBUG)
    # setting coloredlogs
    coloredlogs.install(fmt=console_format, level=console_level,
                        sys=sys.stdout)


def seed_everything(seed: int) -> int:
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True



def pretty_print_config(config: Dict[str, Any]):
    print(json.dumps(config, indent=4, sort_keys=True))


def _load_yml(config_file: pathtype):
    """Helper function to read a yaml file"""
    with open(config_file, "rb") as f:
        config = yaml.safe_load(f)
    return config


def load_config(config_file: pathtype,
                ext_vars: Dict[str, Any] = None,
                use_omega_conf: bool = False) -> Dict[str, Any]:
    """Helper function to read a config file"""

    if not os.path.exists(config_file):
        raise FileNotFoundError(f'{config_file} file do not exists')
    config = _load_yml(config_file)
    return config


def create_dir_from_path(path: str):
    path = Path(path)
    parent_dir: Path = path.parents[0]
    parent_dir.mkdir(parents=True, exist_ok=True)


def write_json(json_data, path: str):
    create_dir_from_path(path)
    with open(path, 'w') as f:
        json.dump(json_data, f, indent=4, sort_keys=True)


def load_json(path: str) -> Dict[str, Any]:
    with open(path) as f:
        json_data = json.load(f)
    return json_data


def load_experiment_input(confpath: str) -> NamedTuple:
    config = load_config(confpath)
    exp_name = os.path.basename(confpath).split('.')[0]
    return ExperimentInputData(confpath, config, exp_name)


def get_filename(path: str) -> str:
    return os.path.basename(path).split('.')[0]

def get_allowed_columns(cols: List[str]) -> List[str]:
    return [f for f in cols if f not in ('target', 'id')]