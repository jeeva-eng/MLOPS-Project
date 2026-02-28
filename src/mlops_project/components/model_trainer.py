import pandas as pd
import os
import joblib

from src.mlops_project import logger
from src.mlops_project.entity.config_entity import ModelTrainerConfig

from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):

        # =========================
        # Load Data
        # =========================
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)

        train_x = train_data.drop([self.config.target_columns], axis=1)
        test_x = test_data.drop([self.config.target_columns], axis=1)

        train_y = train_data[self.config.target_columns]
        test_y = test_data[self.config.target_columns]

        # =========================
        # Define Models
        # =========================
        models = {
            "LinearRegression": LinearRegression(),

            "ElasticNet": ElasticNet(
                alpha=self.config.alpha,
                l1_ratio=self.config.l1_ratio,
                random_state=42
            ),

            "RandomForest": RandomForestRegressor(
                **self.config.random_forest_params
            ),

            "GradientBoosting": GradientBoostingRegressor(
                **self.config.gradient_boosting_params
            )
        }

        model_report = {}

        # =========================
        # Train & Evaluate
        # =========================
        for model_name, model in models.items():

            logger.info(f"Training {model_name}...")

            model.fit(train_x, train_y)
            preds = model.predict(test_x)

            score = r2_score(test_y, preds)
            model_report[model_name] = score

            logger.info(f"{model_name} R2 Score: {score}")

        # =========================
        # Select Best Model
        # =========================
        best_model_name = max(model_report, key=model_report.get)
        best_model = models[best_model_name]
        best_score = model_report[best_model_name]

        logger.info(f"Best Model Found: {best_model_name}")
        logger.info(f"Best Model Score: {best_score}")

        # =========================
        # Save Best Model as model.joblib
        # =========================
        os.makedirs(self.config.root_dir, exist_ok=True)

        best_model_path = os.path.join(
            self.config.root_dir,
            "model_trainer.joblib"
        )

        

        joblib.dump(best_model, best_model_path)

        logger.info(f"Best model saved at {best_model_path}")

        # =========================
        # (Optional) Save All Models
        # =========================
        for model_name, model in models.items():
            model_path = os.path.join(
                self.config.root_dir,
                f"{model_name}.joblib"
            )
            joblib.dump(model, model_path)

        logger.info("All models saved successfully.")

        return best_model_name, best_score