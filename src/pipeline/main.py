from dataclasses import dataclass
import os
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
from src.exception import CustomException
from src.logger import logging
import sys
from src.components.data_transformations import DataTransformations
from src.components.data_ingestion import DataIngestionPipeline
from src.components.utils import DatasetUtils
from src.components.model_trainer import ModelTrainer


@dataclass
class PipelineConfig:
    """Configuration for data pipeline"""
    batch_size: int = 1
    extract_dir: Path = Path("nih_images") / "images"
    meta_csv_path: Path = Path("dataset") / "Data_Entry_2017_v2020.csv"
    img_size: tuple = (128, 128)
    train_batch_size: int = 32
    epochs: int = 5
    links: List[str] = None

    def __post_init__(self):
        self.links = [
            'https://nihcc.box.com/shared/static/vfk49d74nhbxq3nqjg0900w5nvkorp5c.gz',
            # 'https://nihcc.box.com/shared/static/i28rlmbvmfjbl8p2n3ril0pptcmcu9d1.gz',
            # 'https://nihcc.box.com/shared/static/f1t00wrtdk94satdfb9olcolqx20z2jp.gz',
            # 'https://nihcc.box.com/shared/static/0aowwzs5lhjrceb3qp67ahp0rd1l1etg.gz',
            # 'https://nihcc.box.com/shared/static/v5e3goj22zr6h8tzualxfsqlqaygfbsn.gz',
            # 'https://nihcc.box.com/shared/static/asi7ikud9jwnkrnkj99jnpfkjdes7l6l.gz',
            # 'https://nihcc.box.com/shared/static/jn1b4mw4n6lnh74ovmcjb8y48h8xj07n.gz',
            # 'https://nihcc.box.com/shared/static/tvpxmn7qyrgl0w8wfh9kqfjskv6nmm1j.gz',
            # 'https://nihcc.box.com/shared/static/upyy3ml7qdumlgk2rfcvlb9k6gvqq2pj.gz',
            # 'https://nihcc.box.com/shared/static/l6nilvfa9cg3s28tqv1qc1olm3gnz54p.gz',
            # 'https://nihcc.box.com/shared/static/hhq8fkdgvcari67vfhs7ppg2w6ni4jze.gz',
            # 'https://nihcc.box.com/shared/static/ioqwiy20ihqwyr8pf4c24eazhh281pbu.gz'
        ]


class DataPipeline:
    """Main pipeline for handling data processing and model training"""
    
    def __init__(self, config: PipelineConfig):
        """Initialize pipeline with configuration"""
        self.config = config
        self.transformer = DataTransformations(
            meta_csv_path=str(config.meta_csv_path)
        )
        self.ingestion = DataIngestionPipeline(
            img_size=config.img_size,
            batch_size=config.train_batch_size
        )
        self.model_trainer = ModelTrainer(
            img_size=config.img_size,
            epochs=config.epochs
        )
        
    def process_batch(self, 
                     batch_start: int, 
                     train_df: pd.DataFrame, 
                     test_df: pd.DataFrame, 
                     labels: List[str]) -> None:
        """Process a single batch of data"""
        try:
            batch_links = self.config.links[batch_start:batch_start + self.config.batch_size]
            batch_files = []
            batch_num = batch_start//self.config.batch_size + 1

            # Download batch
            for idx, link in enumerate(batch_links):
                filename = f"images_{batch_start+idx+1:02d}.tar.gz"
                DatasetUtils.download_file(link, filename)
                batch_files.append(filename)

            # Extract batch
            for tar_file in batch_files:
                DatasetUtils.extract_tar_gz(tar_file, self.config.extract_dir)
            logging.info(f"Batch {batch_num} extracted to {self.config.extract_dir}")

            # Create generators for current batch
            generators = self.ingestion.create_generators(
                train_df=train_df,
                test_df=test_df,
                image_dir=self.config.extract_dir,
                batch_num=batch_num,
                labels=labels
            )

            if generators:
                # Train model on current batch
                self.model_trainer.train_batch(
                    generators['train_generator'],
                    generators['valid_generator'],
                    batch_num
                )

            # Cleanup batch
            DatasetUtils.cleanup_files(batch_files)
            DatasetUtils.cleanup_files([self.config.extract_dir])
            logging.info(f"Batch {batch_num} processed and cleaned up")

        except Exception as e:
            logging.error(f"Error processing batch {batch_num}")
            raise CustomException(e, sys)

    def run(self) -> None:
        """Execute the complete pipeline"""
        try:
            # Process metadata
            train_df, test_df, labels = self.transformer.process_pipeline()
            logging.info("Metadata processed successfully")

            # Create extraction directory
            os.makedirs(self.config.extract_dir, exist_ok=True)

            # Process batches
            for batch_start in range(0, len(self.config.links), self.config.batch_size):
                self.process_batch(batch_start, train_df, test_df, labels)

            # Save final model
            self.model_trainer.save_model()
            logging.info("Pipeline completed successfully")

        except Exception as e:
            logging.error("Pipeline failed")
            raise CustomException(e, sys)

def main():
    """Main entry point"""
    try:
        # Initialize and run pipeline
        config = PipelineConfig()
        pipeline = DataPipeline(config)
        pipeline.run()

    except Exception as e:
        logging.error("Failed to execute pipeline")
        raise CustomException(e, sys)

if __name__ == "__main__":
    main()
