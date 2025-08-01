import os
import pandas as pd
import requests
import zipfile
import io
from typing import Tuple, List
from .config import PipelineConfig

class DataLoader:
    """Handles downloading and loading of MovieLens dataset"""
    
    def __init__(self, download_dir: str = None):
        self.download_dir = download_dir or PipelineConfig.DOWNLOAD_DIR
        self.movielens_url = PipelineConfig.MOVIELENS_100K_URL
        self.data_root_dir = PipelineConfig.DATA_ROOT_DIR
        
    def download_and_extract_data(self) -> str:
        """
        Downloads MovieLens 100K dataset and extracts it.
        
        Returns:
            str: Path to the extracted data directory
        """
        os.makedirs(self.download_dir, exist_ok=True)
        
        zip_file_path = os.path.join(self.download_dir, os.path.basename(self.movielens_url))

        # Check if data already exists
        if os.path.exists(self.data_root_dir) and os.path.isdir(self.data_root_dir) and \
           os.path.exists(os.path.join(self.data_root_dir, "u.data")):
            print(f"Data already extracted to {self.data_root_dir}. Skipping download and extraction.")
            return self.data_root_dir

        print(f"Downloading dataset from: {self.movielens_url}")
        response = requests.get(self.movielens_url, stream=True)
        response.raise_for_status()

        with open(zip_file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded to: {zip_file_path}")

        print(f"Extracting data to: {self.download_dir}")
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(self.download_dir)
        print("Extraction complete.")

        # Find the correct data path
        if os.path.exists(os.path.join(self.download_dir, "ml-100k", "u.data")):
            final_data_path = os.path.join(self.download_dir, "ml-100k")
        elif os.path.exists(os.path.join(self.download_dir, "ml-100k", "ml-100k", "u.data")):
            final_data_path = os.path.join(self.download_dir, "ml-100k", "ml-100k")
        else:
            raise FileNotFoundError("Could not locate u.data after extraction.")

        print(f"Identified base data directory: {final_data_path}")
        return final_data_path

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Loads the MovieLens dataset files.
        
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: ratings_df, movies_df, users_df
        """
        base_path = self.download_and_extract_data()
        
        # Load ratings data
        ratings_path = os.path.join(base_path, "u.data")
        ratings_df = pd.read_csv(ratings_path, sep='\t', header=None,
                                 names=['UserID', 'MovieID', 'Rating', 'Timestamp'])
        print(f"Loaded {len(ratings_df)} ratings entries.")

        # Load movies data
        movies_path = os.path.join(base_path, "u.item")
        movies_df = pd.read_csv(movies_path, sep='|', encoding='latin-1', header=None,
                                names=['MovieID', 'Title', 'ReleaseDateStr', 'VideoReleaseDate', 'IMDbURL',
                                       'unknown', 'Action', 'Adventure', 'Animation', "Children's", 'Comedy',
                                       'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
                                       'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'])
        print(f"Loaded {len(movies_df)} movies entries.")

        # Load users data
        users_path = os.path.join(base_path, "u.user")
        users_df = pd.read_csv(users_path, sep='|', header=None,
                               names=['UserID', 'Age', 'Gender', 'Occupation', 'Zip-code'])
        print(f"Loaded {len(users_df)} users entries.")

        return ratings_df, movies_df, users_df 