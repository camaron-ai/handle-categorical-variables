import logging
from src import util
from src.process_cat import STRATEGY_DISPATCHER
from src.run_experiment import run_experiment_from_config
import ray
import itertools


ray.init(num_cpus=1)
logger = logging.getLogger(__name__)


GLOBAL_MODELS = [
'confs/models/xgb.yml',
'confs/models/logreg.yml'
]

STRATEGY_NAMES = list(STRATEGY_DISPATCHER.keys())

@ray.remote
def run_parallel(dataset_name: str, confpath: str):
    run_experiment_from_config(dataset_name, confpath)


if __name__ == '__main__':
    util.setup_logging()
    requests = [
        run_parallel.remote(strategy, confpath)
        for strategy, confpath in itertools.product(STRATEGY_NAMES, GLOBAL_MODELS)]
    ray.get(requests)
