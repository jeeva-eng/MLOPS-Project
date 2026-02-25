import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src.mlops_project import logger
from src.mlops_project.entity.config_entity import DataTransformationConfig


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def train_test_spliting(self):

        data = pd.read_csv(self.config.data_path)

        train, test = train_test_split(data, test_size=0.25, random_state=42)

        train.to_csv(os.path.join(self.config.root_dir, "train.csv"), index=False)
        test.to_csv(os.path.join(self.config.root_dir, "test.csv"), index=False)

        logger.info("Splitted data into training and test sets")
        logger.info(f"Train shape: {train.shape}")
        logger.info(f"Test shape: {test.shape}")

        return train, test


    def get_data_transformer_object(self, train_df):

        target_column = self.config.target_column

        input_features = train_df.drop(columns=[target_column])

        numerical_columns = input_features.select_dtypes(include=["int64", "float64"]).columns.tolist()

        logger.info(f"Numerical Columns: {numerical_columns}")

        num_pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler())
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num_pipeline", num_pipeline, numerical_columns)
            ]
        )

        return preprocessor


    def initiate_data_transformation(self):

        train_df, test_df = self.train_test_spliting()

        target_column = self.config.target_column

        preprocessor = self.get_data_transformer_object(train_df)

        input_train = train_df.drop(columns=[target_column])
        target_train = train_df[target_column]

        input_test = test_df.drop(columns=[target_column])
        target_test = test_df[target_column]

        input_train_arr = preprocessor.fit_transform(input_train)
        input_test_arr = preprocessor.transform(input_test)

        # Save preprocessor
        joblib.dump(preprocessor, os.path.join(self.config.root_dir, "preprocessor.pkl"))

        logger.info("Preprocessor saved successfully")

        return (
            input_train_arr,
            target_train,
            input_test_arr,
            target_test
        )