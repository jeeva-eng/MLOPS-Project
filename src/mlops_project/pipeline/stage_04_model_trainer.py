from src.mlops_project.config.configuration import ConfigurationManager
from src.mlops_project.components.model_trainer import ModelTrainer
from src.mlops_project import logger
from pathlib import Path


STAGE_NAME="Model Trainer Stage"

class ModelTrainerTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
         config=ConfigurationManager()
         model_trainer_config=config.get_model_trainer_config()
         model_trainer_config=ModelTrainer(config=model_trainer_config)
         model_trainer_config.train()

if __name__ =='__main__':
    try:
        logger.info(f">>>> syge{STAGE_NAME} started <<<<<")
        obj=ModelTrainerTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<<\n\nx========x")
    except Exception as e:
        logger.exception(0)
    raise e            