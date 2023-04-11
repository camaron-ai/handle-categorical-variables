import os

WORKDIR = os.path.dirname(os.path.abspath(__file__))
# DEFINING USEFUL PATH
MLFLOW_DEFAULT_EXPERIMENT = 'leaderboard'
# RANDOM PARAMTERS
MAIN_SEED = 2022
TARGET_NAME = 'target'

os.environ['WORKDIR'] = WORKDIR
os.environ['MAIN_SEED'] = str(MAIN_SEED)
os.environ['MLFLOW_EXPERIMENT'] = MLFLOW_DEFAULT_EXPERIMENT


CATEGORICAL_VARIABLES = [
 'ps_ind_02_cat',
 'ps_ind_04_cat',
 'ps_ind_05_cat',
 'ps_car_01_cat',
 'ps_car_02_cat',
 'ps_car_03_cat',
 'ps_car_04_cat',
 'ps_car_05_cat',
 'ps_car_06_cat',
 'ps_car_07_cat',
 'ps_car_08_cat',
 'ps_car_09_cat',
 'ps_car_10_cat',
 'ps_car_11_cat']


VARIABLES_TO_MIN_MAX_SCALE = ['ps_ind_01',
 'ps_ind_03',
 'ps_ind_14',
 'ps_ind_15',
 'ps_reg_01',
 'ps_reg_02',
 'ps_car_11',
 'ps_car_15',
 'ps_calc_01',
 'ps_calc_02',
 'ps_calc_03',
 'ps_calc_04',
 'ps_calc_05',
 'ps_calc_06',
 'ps_calc_07',
 'ps_calc_08',
 'ps_calc_09',
 'ps_calc_10',
 'ps_calc_11',
 'ps_calc_12',
 'ps_calc_13',
 'ps_calc_14']


class EnviromentManager:
    def __init__(
        self,
        workdir: str,
        main_seed: int,
        mlflow_experiment: str = None,
        ):
        self.workdir = workdir
        self.main_seed = main_seed
        self._mlflow_experiment = mlflow_experiment


    @property
    def data_dir(self):
        return os.path.join(self.workdir, 'data')
    
    @property
    def raw_dir(self):
        return os.path.join(self.data_dir, 'raw')

    def interim_dir(self, set_type: str):
        return os.path.join(self.data_dir, 'interim', set_type)
    
    @property
    def raw_train_path(self):
        return os.path.join(self.raw_dir, 'train.csv')
    
    @property
    def raw_test_path(self):
        return os.path.join(self.raw_dir, 'test.csv')


    @property 
    def clean_train_path(self):
        return os.path.join(self.data_dir, 'clean', 'data.parquet')
        
    @property
    def experiment_output_dir(self):
        return os.path.join(self.workdir, 'experiment_output')

    @property
    def proccesed_dir(self):
        return os.path.join(
            self.data_dir,
            'processed',
            )

    @property    
    def cv_path(self):
        return os.path.join(self.data_dir, 'split', 'split_data.csv')

    @property
    def features_processed_dir(self):
        return os.path.join(self.proccesed_dir, 'features')

    def format_experiment_path(self, experiment_name: str):
        return os.path.join(
            self.experiment_output_dir,
            experiment_name,
            )

    def format_features_dir(self, dataset_name: str):
        return os.path.join(self.features_processed_dir, dataset_name)
    
    @classmethod
    def from_enviroment(cls):
        workdir = os.environ['WORKDIR'] 
        main_seed = os.environ['MAIN_SEED']
        mlflow_experiment = os.environ['MLFLOW_EXPERIMENT']
        return cls(workdir, main_seed, mlflow_experiment)



# DATA RELATED CONFIG
if __name__ == '__main__':
    test_manager = EnviromentManager.from_enviroment(use_sample=False)
    assert test_manager.workdir == WORKDIR


