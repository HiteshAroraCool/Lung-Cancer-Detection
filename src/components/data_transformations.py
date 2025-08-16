from src.exception import CustomException
from src.logger import logging
import sys
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split

class DataTransformations:
    def load_metadata_file(META_CSV_PATH):
        # loading metadata file
        try:
            # Load NIH metadata CSV
            data_entry_df = pd.read_csv(META_CSV_PATH)
            logging.info("Meta Dataset Loaded Successfully")
        except Exception as e:
            logging.error("Fail to load metadata file. stopping all excuation!!!")
            raise CustomException(e, sys)
        return data_entry_df
    
    def one_hot_encoding(data_entry_df):
        try:
            data_entry_df.replace('No Finding', '', inplace=True)
            ### Applying One-Hot Encoding
            multiple_list = data_entry_df['Finding Labels'].str.split('|')
            flattened_list = [item for sublist in multiple_list for item in sublist]
            label_count = dict(Counter(flattened_list).most_common())
            all_labels = [x for x in label_count if len(x)>0]
            logging.info(f"Found Number of Labels {len(all_labels)} \n{all_labels}")
            for c_label in all_labels:
                if len(c_label)>1: # leave out empty labels
                    data_entry_df[c_label] = data_entry_df['Finding Labels'].map(lambda finding: 1.0 if c_label in finding else 0)
            logging.info("One-Hot Encording Successfully Loaded")

            # sample weights to initialise
            sample_weights = data_entry_df['Finding Labels'].map(lambda x: len(x.split('|')) if len(x)>0 else 0).values + 4e-2
            sample_weights /= sample_weights.sum()
            data_entry_df = data_entry_df.sample(40000, weights=sample_weights)
            data_entry_df['disease_vec'] = data_entry_df.apply(lambda x: [x[all_labels[1:]].values], axis=1).map(lambda x: x[0])

            train, test = train_test_split(data_entry_df, test_size=0.25, random_state=42, stratify=data_entry_df['Finding Labels'].map(lambda x: x[:4]))

        except Exception as e:
            logging.error("Fail to Transform data. stopping all excuation!!!")
            raise CustomException(e, sys)
        
    
        return train, test, all_labels