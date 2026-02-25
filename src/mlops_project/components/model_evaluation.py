import pandas as pd
import numpy as np
import joblib
from urllib.parse import urlparse
from pathlib import Path

import mlflow
import mlflow.sklearn

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.mlops_project import logger
from src.mlops_project.entity.config_entity import ModelEvaluationConfig
from src.mlops_project.utils.common import save_json


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    # ---------------------------
    # Metric Calculation
    # ---------------------------
    def eval_metrics(self, actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2

    # ---------------------------
    # MLflow Logging + Evaluation
    # ---------------------------
    def log_into_mlflow(self):

        try:
            logger.info("Starting Model Evaluation...")

            # Load test data
            test_data = pd.read_csv(self.config.test_data_path)
            test_x = test_data.drop([self.config.target_columns], axis=1)
            test_y = test_data[self.config.target_columns]

            # Set MLflow registry
            mlflow.set_registry_uri(self.config.mlflow_uri)
            tracking_url_type_store = urlparse(
                mlflow.get_tracking_uri()
            ).scheme

            # Directory containing saved models
            model_dir = Path(self.config.model_path).parent

            best_score = -1
            best_model_name = None
            best_metrics = {}

            # Loop through all models
            for model_file in model_dir.glob("*.joblib"):

                model_name = model_file.stem
                model = joblib.load(model_file)

                logger.info(f"Evaluating Model: {model_name}")

                with mlflow.start_run(run_name=model_name):

                    # Predict
                    predicted = model.predict(test_x)

                    # Metrics
                    rmse, mae, r2 = self.eval_metrics(test_y, predicted)

                    logger.info(f"{model_name} -> RMSE: {rmse}")
                    logger.info(f"{model_name} -> MAE: {mae}")
                    logger.info(f"{model_name} -> R2: {r2}")

                    # Log parameters (if available)
                    for model_key, param_dict in self.config.all_params.items():
                        for param, value in param_dict.items():
                            mlflow.log_param(f"{model_key}_{param}", value)

                    # Log metrics
                    mlflow.log_metric("rmse", rmse)
                    mlflow.log_metric("mae", mae)
                    mlflow.log_metric("r2", r2)

                    # Log model
                    if tracking_url_type_store != "file":
                        mlflow.sklearn.log_model(
                            model,
                            "model",
                            registered_model_name=model_name
                        )
                    else:
                        mlflow.sklearn.log_model(model, "model")

                # Track best model
                if r2 > best_score:
                    best_score = r2
                    best_model_name = model_name
                    best_metrics = {
                        "best_model": model_name,
                        "rmse": rmse,
                        "mae": mae,
                        "r2": r2
                    }

            # Save best metrics locally
            save_json(
                path=Path(self.config.metric_file_name),
                data=best_metrics
            )

            logger.info(f"Best Model in Evaluation: {best_model_name}")
            logger.info("Model Evaluation Completed Successfully")

        except Exception as e:
            logger.error("Error during model evaluation")
            raise e