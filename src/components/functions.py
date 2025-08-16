from src.exception import CustomException
from src.logger import logging
import sys
import requests
import tarfile
import shutil
from tqdm import tqdm
import os

class DatasetUtils:
    # ====== FUNCTIONS ======
    @staticmethod
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
    
    @staticmethod
    def extract_tar_gz(tar_path, extract_to):
        logging.info(f"Extracting {tar_path} ...")
        with tarfile.open(tar_path, "r:gz") as tar:
            members = tar.getmembers()
            for member in tqdm(members, desc=f"Extracting {os.path.basename(tar_path)}", unit="file"):
                tar.extract(member, path=extract_to)

    @staticmethod
    def cleanup_files(paths):
        for p in paths:
            if os.path.isdir(p):
                shutil.rmtree(p)
            elif os.path.isfile(p):
                os.remove(p)
            logging.info(f"{p} Sucessfully Removed")
