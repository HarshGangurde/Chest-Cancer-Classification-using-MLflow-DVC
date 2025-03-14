import os
import zipfile
import gdown
from cnnClassifier import logger
from cnnClassifier.utils.common import get_size
from cnnClassifier.entity.config_entity import DataIngestionConfig

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self) -> str:
        """
        Fetch data from the URL and download it as a ZIP file.
        """
        try:
            dataset_url = self.config.source_URL
            zip_download_dir = self.config.local_data_file
            os.makedirs(os.path.dirname(zip_download_dir), exist_ok=True)

            logger.info(f"Downloading data from {dataset_url} into file {zip_download_dir}")

            # Ensure the correct format for Google Drive link
            file_id = dataset_url.split("/")[-2]
            gdrive_url = f"https://drive.google.com/uc?id=1uVKWc8-zuHAvEp3Wv7vsTn9nBtsXEN-7"

            # Download the file using gdown
            gdown.download(gdrive_url, zip_download_dir, quiet=False)

            logger.info(f"Downloaded data from {dataset_url} into file {zip_download_dir}")

        except Exception as e:
            logger.error(f"Error downloading file: {e}")
            raise e

    def extract_zip_file(self):
        """
        Extracts the zip file into the data directory.
        """
        try:
            unzip_path = self.config.unzip_dir
            os.makedirs(unzip_path, exist_ok=True)

            logger.info(f"Extracting {self.config.local_data_file} to {unzip_path}")

            with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
                zip_ref.extractall(unzip_path)

            logger.info(f"Extraction completed: {unzip_path}")

        except Exception as e:
            logger.error(f"Error extracting file: {e}")
            raise e
