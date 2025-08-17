from typing import Tuple, List, Dict
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from src.exception import CustomException
from src.logger import logging
import sys

class DataTransformations:
    """
    A class to handle data transformations for the lung cancer detection project.
    
    Attributes:
        meta_csv_path (str): Path to the metadata CSV file
        random_state (int): Random state for reproducibility
        test_size (float): Proportion of dataset to include in the test split
    """

    def __init__(self, meta_csv_path: str, random_state: int = 42, test_size: float = 0.25):
        self.meta_csv_path = meta_csv_path
        self.random_state = random_state
        self.test_size = test_size
        self.data_df = None
        self.all_labels = None

    def load_metadata_file(self) -> pd.DataFrame:
        """
        Load and validate the metadata CSV file.
        
        Returns:
            pd.DataFrame: Loaded metadata DataFrame
        
        Raises:
            CustomException: If file loading fails
        """
        try:
            self.data_df = pd.read_csv(self.meta_csv_path)
            logging.info(f"Metadata loaded successfully: {self.data_df.shape} records")
            return self.data_df
        except Exception as e:
            logging.error("Failed to load metadata file")
            raise CustomException(e, sys)

    def _extract_labels(self) -> List[str]:
        """
        Extract unique labels from the Finding Labels column.
        
        Returns:
            List[str]: List of unique disease labels
        """
        try:
            multiple_list = self.data_df['Finding Labels'].str.split('|')
            flattened_list = [item for sublist in multiple_list for item in sublist]
            label_count = dict(Counter(flattened_list).most_common())
            self.all_labels = [x for x in label_count if len(x)>0]
            logging.info(f"Extracted {len(self.all_labels)} unique labels")
            return self.all_labels
        except Exception as e:
            logging.error("Failed to extract labels")
            raise CustomException(e, sys)

    def _compute_sample_weights(self) -> np.ndarray:
        """
        Compute sample weights based on disease occurrences.
        
        Returns:
            np.ndarray: Array of sample weights
        """
        weights = (self.data_df['Finding Labels']
                  .map(lambda x: len(x.split('|')) if len(x)>0 else 0)
                  .values + 4e-2)
        return weights / weights.sum()

    def transform_data(self, sample_size: int = 40000) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
        """
        Transform the data including one-hot encoding and train-test split.
        
        Args:
            sample_size (int): Number of samples to use
        
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, List[str]]: Train data, test data, and labels
        """
        try:
            # Replace 'No Finding' with empty string
            self.data_df.replace('No Finding', '', inplace=True)
            
            # Extract labels and perform one-hot encoding
            all_labels = self._extract_labels()
            for label in all_labels[1:]:  # Skip empty label
                self.data_df[label] = self.data_df['Finding Labels'].map(
                    lambda finding: 1.0 if label in finding else 0
                )
            
            # Sample data with weights
            weights = self._compute_sample_weights()
            self.data_df = self.data_df.sample(sample_size, weights=weights)
            
            # Create disease vectors
            self.data_df['disease_vec'] = self.data_df.apply(
                lambda x: [x[all_labels[1:]].values], 
                axis=1
            ).map(lambda x: x[0])
            
            # Split data
            train, test = train_test_split(
                self.data_df,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=self.data_df['Finding Labels'].map(lambda x: x[:4])
            )
            
            logging.info(f"Data transformed successfully. Train: {train.shape}, Test: {test.shape}")
            return train, test, all_labels
            
        except Exception as e:
            logging.error("Failed to transform data")
            raise CustomException(e, sys)

    def process_pipeline(self) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
        """
        Execute the complete data processing pipeline.
        
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, List[str]]: Processed train data, test data, and labels
        """
        self.load_metadata_file()
        return self.transform_data()