from src.exception import CustomException
from src.logger import logging
import sys
import requests
import tarfile
import shutil
from tqdm import tqdm
from typing import List, Union
from pathlib import Path

class DatasetUtils:
    """Utility class for handling dataset operations like download, extraction, and cleanup."""

    # ====== FUNCTIONS ======
    @staticmethod
    def download_file(url: str, save_path: Union[str, Path], chunk_size: int = 1024) -> None:
        """
        Download a file from URL with progress bar.
        
        Args:
            url: Source URL
            save_path: Destination path
            chunk_size: Size of chunks to download
        """
        try:
            with requests.get(url, stream=True) as response:
                response.raise_for_status()
                total_size = int(response.headers.get('content-length', 0))
                
                with (tqdm(total=total_size, unit='B', unit_scale=True, desc=str(save_path)) as pbar,
                      open(save_path, 'wb') as file):
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            file.write(chunk)
                            pbar.update(len(chunk))
            logging.info(f"Successfully downloaded: {save_path}")
        except Exception as e:
            logging.error(f"Failed to download file: {url}")
            raise CustomException(e, sys)
        
    @staticmethod
    def extract_tar_gz(tar_path: Union[str, Path], extract_to: Union[str, Path]) -> None:
        """
        Extract tar.gz file to specified directory.
        
        Args:
            tar_path: Path to tar.gz file
            extract_to: Extraction destination
        """
        try:
            logging.info(f"Extracting {tar_path}")
            with tarfile.open(tar_path, "r:gz") as tar:
                members = tar.getmembers()
                for member in tqdm(members, desc=f"Extracting {Path(tar_path).name}", unit="file"):
                    tar.extract(member, path=extract_to)
            logging.info(f"Successfully extracted to: {extract_to}")
        except Exception as e:
            logging.error(f"Failed to extract: {tar_path}")
            raise CustomException(e, sys)
        
    @staticmethod
    def cleanup_files(paths: List[Union[str, Path]]) -> None:
        """
        Remove files and directories.
        
        Args:
            paths: List of paths to clean up
        """
        try:
            for path in paths:
                path = Path(path)
                if path.is_dir():
                    shutil.rmtree(path)
                elif path.is_file():
                    path.unlink()
                logging.info(f"Successfully removed: {path}")
        except Exception as e:
            logging.error(f"Failed to remove paths: {paths}")
            raise CustomException(e, sys)
