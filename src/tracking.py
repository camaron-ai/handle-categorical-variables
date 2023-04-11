import mlflow
import os
from typing import Dict, Any

def track_experiment(run_name: str, 
                     mlflow_experiment: str,
                     parameters: Dict[str, Any],
                     scores: Dict[str, float],
                     artifacts_dir: str = None,
                     tags: Dict[str, str] = None,
                     ):
    mlflow.set_experiment(mlflow_experiment)

    with mlflow.start_run(run_name=run_name):
        if tags:
            mlflow.set_tags(tags)
        # log parameters
        mlflow.log_params(parameters)
        # log metrics
        mlflow.log_metrics(scores)
        # log all artifacts
        if artifacts_dir:
            mlflow.log_artifacts(artifacts_dir)
