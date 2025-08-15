## File to Download Dataset(https://nihcc.app.box.com/v/ChestXray-NIHCC/folder/36938765345) in batches.

import os
import tarfile
import shutil
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from src.exception import CustomException
from src.logger import logging
import sys
import requests

# ====== FUNCTIONS ======

def download_file(url, save_path):
    # Stream download
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total_size = int(r.headers.get('content-length', 0))
        block_size = 1024  # 1 KB chunks
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=save_path) as pbar:
            with open(save_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=block_size):
                    if chunk:  # filter out keep-alive chunks
                        f.write(chunk)
                        pbar.update(len(chunk))

def extract_tar_gz(tar_path, extract_to):
    logging.info(f"Extracting {tar_path} ...")
    with tarfile.open(tar_path, "r:gz") as tar:
        members = tar.getmembers()
        for member in tqdm(members, desc=f"Extracting {os.path.basename(tar_path)}", unit="file"):
            tar.extract(member, path=extract_to)

def cleanup_files(paths):
    for p in paths:
        if os.path.isdir(p):
            shutil.rmtree(p)
        elif os.path.isfile(p):
            os.remove(p)
        logging.info(f"{p} Sucessfully Removed")

# ====== MAIN ======
if __name__ == "__main__":

    # ====== CONFIG ======
    try:
        META_CSV = os.path.join("dataset", "Data_Entry_2017_v2020.csv")
        # Load NIH metadata CSV
        meta_df = pd.read_csv(META_CSV)
        logging.info("Meta Dataset Loaded Successfully")

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
                    download_file(link, filename)
                    batch_files.append(filename)

                # Extract batch
                for tar_file in batch_files:
                    extract_tar_gz(tar_file, EXTRACT_DIR)

                logging.info(f"Batch {batch_start//BATCH_SIZE + 1} ready in {EXTRACT_DIR}")

                ## training here with try:

                # Cleanup batch
                cleanup_files(batch_files)      # remove tar.gz
                cleanup_files([EXTRACT_DIR])    # remove extracted images
                logging.info(f"Batch {batch_start//BATCH_SIZE + 1} cleaned up.")

            except Exception as e:
                logging.error(f"Error processing batch {batch_start//BATCH_SIZE + 1}")
                raise CustomException(e, sys)

        logging.info("All batches processed successfully.")

    except Exception as e:
        logging.error("Failed to process dataset")
        raise CustomException(e, sys)