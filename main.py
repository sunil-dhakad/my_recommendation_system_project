from src.exception import CustomException
from src.logger import logging
from src.config.configuration import AppConfiguration
from src.components.data_ingestion import DataIngestion
import os
from src.pipeline.training_pipeline import TrainingPipeline


def main():
    try:
        pipeline = TrainingPipeline()
        pipeline.start_training_pipeline()

    except Exception as e:
            logging.error(f"{e}")
            print(e)


if __name__ == "__main__":
     main()