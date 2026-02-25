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

    def eval_metrics(self, actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2

    def log_into_mlflow(self):

        try:
            logger.info("Starting Model Evaluation...")

            test_data = pd.read_csv(self.config.test_data_path)
            model = joblib.load(self.config.model_path)

            test_x = test_data.drop([self.config.target_columns], axis=1)
            test_y = test_data[self.config.target_columns]

            mlflow.set_registry_uri(self.config.mlflow_uri)

            tracking_url_type_store = urlparse(
                mlflow.get_tracking_uri()
            ).scheme

            with mlflow.start_run():

                predicted = model.predict(test_x)

                rmse, mae, r2 = self.eval_metrics(test_y, predicted)

                logger.info(f"RMSE: {rmse}")
                logger.info(f"MAE: {mae}")
                logger.info(f"R2: {r2}")

                # Save metrics locally
                scores = {"rmse": rmse, "mae": mae, "r2": r2}
                save_json(
                    path=Path(self.config.metric_file_name),
                    data=scores
                )

                # Log parameters
                mlflow.log_params(self.config.all_params)

                # Log metrics
                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("mae", mae)
                mlflow.log_metric("r2", r2)

                # Log model
                if tracking_url_type_store != "file":
                    mlflow.sklearn.log_model(
                        model,
                        "model",
                        registered_model_name="WineQualityModel"
                    )
                else:
                    mlflow.sklearn.log_model(model, "model")

            logger.info("Model Evaluation Completed Successfully")

        except Exception as e:
            logger.error("Error during model evaluation")
            raise e