## File to Download Dataset(https://nihcc.app.box.com/v/ChestXray-NIHCC/folder/36938765345) in batches.

import os
import urllib.request
import tarfile
import shutil
from pathlib import Path
from tqdm import tqdm
import pandas as pd

# ====== FUNCTIONS ======

def download_file(url, save_path):
    print(f"Downloading {save_path} ...")
    urllib.request.urlretrieve(url, save_path)

def extract_tar_gz(tar_path, extract_to):
    print(f"Extracting {tar_path} ...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=extract_to)

def cleanup_files(paths):
    for p in paths:
        if os.path.isdir(p):
            shutil.rmtree(p)
        elif os.path.isfile(p):
            os.remove(p)

# ====== MAIN ======
if __name__ == "__main__":

    # ====== CONFIG ======
    META_CSV = "dataset\Data_Entry_2017_v2020.csv"
    # Load NIH metadata CSV
    meta_df = pd.read_csv(META_CSV)

    BATCH_SIZE = 1  # number of tar.gz files to download at once
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

    # ====== MAIN BATCH LOOP ======
    os.makedirs(EXTRACT_DIR, exist_ok=True)

    for batch_start in range(0, len(LINKS), BATCH_SIZE):
        batch_links = LINKS[batch_start:batch_start + BATCH_SIZE]
        batch_files = []

        # 1. Download batch
        for idx, link in enumerate(tqdm(batch_links, desc="Downloading", unit="file")):
                filename = f"images_{batch_start+idx+1:02d}.tar.gz"
                download_file(link, filename)
                batch_files.append(filename)

        # 2. Extract
        for tar_file in tqdm(batch_files, desc="Extracting", unit="file"):
            extract_tar_gz(tar_file, EXTRACT_DIR)

        print(f"Batch {batch_start//BATCH_SIZE + 1} ready in {EXTRACT_DIR}")

        # === TRAINING ===
        # train_on_directory(EXTRACT_DIR)

        # 3. Cleanup batch to save space
        cleanup_files(batch_files)      # remove tar.gz
        cleanup_files([EXTRACT_DIR])    # remove extracted images
        print(f"Batch {batch_start//BATCH_SIZE + 1} cleaned up.\n")

    print("All batches processed.")