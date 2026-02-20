from src.mlops_project import logger
from src.mlops_project.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from src.mlops_project.pipeline.stage_02_data_validation import DataValidationTrainingPipeline

STAGE_NAME="Data Ingestion Stage"

try:
    logger.info(f">>>>>> stage{STAGE_NAME} started <<<<<<")
    data_ingestion=DataIngestionTrainingPipeline()
    data_ingestion.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<<\n\nx==========x")

except Exception as e:
        logger.exception(e)
        raise e


STAGE_NAME="Data Validation Stage"

try:
    logger.info(f">>>>>> stage{STAGE_NAME} started <<<<<<")
    data_ingestion=DataValidationTrainingPipeline()
    data_ingestion.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<<\n\nx==========x")

except Exception as e:
        logger.exception(e)
        raise e