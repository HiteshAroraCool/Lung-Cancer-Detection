## File to Download Dataset(https://nihcc.app.box.com/v/ChestXray-NIHCC/folder/36938765345) in batches.

import os
from pathlib import Path
from src.exception import CustomException
from src.logger import logging
import sys
from src.components.data_transformations import DataTransformations
from src.components.functions import DatasetUtils


# ====== MAIN ======
if __name__ == "__main__":

    # Call data transformations to get train, test, all labels
    transformer = DataTransformations(meta_csv_path=os.path.join("dataset", "Data_Entry_2017_v2020.csv"))
    train, test, labels = transformer.process_pipeline()

    # ====== Batch Rotation ======
    try:
        BATCH_SIZE = 1
        EXTRACT_DIR = Path("nih_images")
        LINKS = [
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

        # Create directory for downloads
        os.makedirs(EXTRACT_DIR, exist_ok=True)

        for batch_start in range(0, len(LINKS), BATCH_SIZE):
            try:
                batch_links = LINKS[batch_start:batch_start + BATCH_SIZE]
                batch_files = []

                # Download batch
                for idx, link in enumerate(batch_links):
                    filename = f"images_{batch_start+idx+1:02d}.tar.gz"
                    DatasetUtils.download_file(link, filename)
                    batch_files.append(filename)

                # Extract batch
                for tar_file in batch_files:
                    DatasetUtils.extract_tar_gz(tar_file, EXTRACT_DIR)
                logging.info(f"Batch {batch_start//BATCH_SIZE + 1} ready in {EXTRACT_DIR}")

                ## training here with try: call train pipeline file

                # Cleanup batch
                DatasetUtils.cleanup_files(batch_files)      # remove tar.gz
                DatasetUtils.cleanup_files([EXTRACT_DIR])    # remove extracted images
                logging.info(f"Batch {batch_start//BATCH_SIZE + 1} cleaned up.")

            except Exception as e:
                logging.error(f"Error processing batch {batch_start//BATCH_SIZE + 1}")
                raise CustomException(e, sys)

        logging.info("All batches processed successfully.")

    except Exception as e:
        logging.error("Failed to process dataset")
        raise CustomException(e, sys)