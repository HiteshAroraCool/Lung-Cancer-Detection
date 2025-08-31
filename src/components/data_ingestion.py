from typing import Tuple, List, Dict
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from pathlib import Path
from src.exception import CustomException
from src.logger import logging
import sys

class DataIngestionPipeline:
    """Handles data ingestion and augmentation in batches."""
    
    def __init__(self, img_size: Tuple[int, int] = (128, 128), batch_size: int = 32):
        self.img_size = img_size
        self.batch_size = batch_size
        self.datagen = self._create_data_generator()
        
    def _create_data_generator(self) -> ImageDataGenerator:
        """Create and configure the ImageDataGenerator."""
        return ImageDataGenerator(
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest',
            validation_split=0.2
        )
    
    def _update_image_paths(self, 
                           df: pd.DataFrame, 
                           image_dir: Path, 
                           batch_num: int) -> pd.DataFrame:
        """Update image paths for the current batch."""
        try:
            # Filter dataframe for current batch images
            batch_images = list(image_dir.glob('*.png'))  #  extension
            image_names = {img.name for img in batch_images}
            
            # Update paths only for images present in the current batch
            batch_df = df[df['Image Index'].isin(image_names)].copy()
            batch_df['image_path'] = batch_df['Image Index'].apply(
                lambda x: str(image_dir / x)
            )
            
            logging.info(f"Batch {batch_num}: Found {len(batch_df)} matching images")
            return batch_df
            
        except Exception as e:
            logging.error(f"Error updating image paths for batch {batch_num}")
            raise CustomException(e, sys)

    def create_generators(self, 
                         train_df: pd.DataFrame,
                         test_df: pd.DataFrame,
                         image_dir: Path,
                         batch_num: int,
                         labels: List[str]) -> Dict:
        """Create data generators for the current batch."""
        try:
            # Update paths for current batch
            batch_train = self._update_image_paths(train_df, image_dir, batch_num)
            batch_test = self._update_image_paths(test_df, image_dir, batch_num)
            
            if len(batch_train) == 0 or len(batch_test) == 0:
                logging.warning(f"No images found for batch {batch_num}")
                return None
            
            # Prepare labels
            for df in [batch_train, batch_test]:
                df['newLabel'] = df['Finding Labels'].str.split('|')
            
            # Create generators
            train_gen = self.datagen.flow_from_dataframe(
                dataframe=batch_train,
                directory=None,
                x_col='image_path',
                y_col='newLabel',
                class_mode='categorical',
                classes=labels,
                target_size=self.img_size,
                color_mode='grayscale',
                batch_size=self.batch_size
            )
            
            valid_gen = self.datagen.flow_from_dataframe(
                dataframe=batch_test,
                directory=None,
                x_col='image_path',
                y_col='newLabel',
                class_mode='categorical',
                classes=labels,
                target_size=self.img_size,
                color_mode='grayscale',
                batch_size=self.batch_size
            )
            
            logging.info(f"Created generators for batch {batch_num}")
            return {
                'train_generator': train_gen,
                'valid_generator': valid_gen,
                'batch_train': batch_train,
                'batch_test': batch_test
            }
            
        except Exception as e:
            logging.error(f"Failed to create generators for batch {batch_num}")
            raise CustomException(e, sys)

if __name__ == "__main__":
    train_df = pd.read_csv("./dataset/Data_Entry_2017_v2020.csv")[:100]
    test_df = pd.read_csv("./dataset/Data_Entry_2017_v2020.csv")[100:200]
    labels = train_df['Finding Labels'].str.split('|').explode().unique().tolist()
    labels = [l for l in labels if l]  # remove empty strings

    generators = DataIngestionPipeline().create_generators(
                train_df=train_df,
                test_df=test_df,
                image_dir=Path("nih_images/images"),
                batch_num=1,
                labels=labels
            )