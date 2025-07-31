import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler, OneHotEncoder
from typing import Tuple, List, Dict, Any
from .config import PipelineConfig

class DataPreprocessor:
    """Handles data preprocessing and feature engineering"""
    
    def __init__(self):
        self.mlb = None
        self.ohe_user_features = None
        self.scaler = None
        self.feature_columns = None
        
    def create_genres_feature(self, movies_df: pd.DataFrame) -> pd.DataFrame:
        """Creates genres feature from movie data"""
        genre_cols_100k = PipelineConfig.GENRE_COLS_100K
        
        movies_df['Genres'] = movies_df[genre_cols_100k].apply(
            lambda row: '|'.join([col for col, val in row.items() if val == 1]), axis=1
        )
        movies_df.drop(columns=['ReleaseDateStr', 'VideoReleaseDate', 'IMDbURL'] + genre_cols_100k, inplace=True)
        
        return movies_df, genre_cols_100k
    
    def extract_release_year(self, movies_df: pd.DataFrame, base_path: str) -> pd.DataFrame:
        """Extracts release year from movie data"""
        movies_path = f"{base_path}/u.item"
        movies_df_for_year = pd.read_csv(movies_path, sep='|', encoding='latin-1', header=None,
                                    names=['MovieID', 'Title', 'ReleaseDateStr', 'VideoReleaseDate', 'IMDbURL'] +
                                           [f'Genre_{i}' for i in range(19)])
        
        movies_df_for_year['ReleaseYear'] = pd.to_datetime(movies_df_for_year['ReleaseDateStr'], format='%d-%b-%Y').dt.year
        
        return movies_df_for_year[['MovieID', 'ReleaseYear']]
    
    def binarize_ratings(self, ratings_df: pd.DataFrame) -> pd.DataFrame:
        """Binarizes ratings into engagement (like/dislike)"""
        ratings_df['Engagement'] = (ratings_df['Rating'] >= PipelineConfig.ENGAGEMENT_THRESHOLD).astype(int)
        print(f"Binarized ratings: {ratings_df['Engagement'].value_counts()}")
        return ratings_df
    
    def merge_datasets(self, ratings_df: pd.DataFrame, movies_df: pd.DataFrame, 
                      users_df: pd.DataFrame, release_year_df: pd.DataFrame) -> pd.DataFrame:
        """Merges all datasets together"""
        df = pd.merge(ratings_df, movies_df, on='MovieID', how='inner')
        df = pd.merge(df, users_df, on='UserID', how='inner')
        df = pd.merge(df, release_year_df, on='MovieID', how='left')
        
        # Fill missing release years with median
        df['ReleaseYear'] = df['ReleaseYear'].fillna(df['ReleaseYear'].median())
        df['ReleaseYear'] = df['ReleaseYear'].astype(float)
        
        print(f"Merged DataFrame shape: {df.shape}")
        return df
    
    def process_genres(self, df: pd.DataFrame, genre_cols: List[str]) -> Tuple[pd.DataFrame, MultiLabelBinarizer]:
        """Processes genres using MultiLabelBinarizer"""
        mlb = MultiLabelBinarizer()
        mlb.fit([genre_cols])
        genre_features = mlb.transform(df['Genres'].apply(lambda x: x.split('|')))
        genre_feature_df = pd.DataFrame(genre_features, columns=mlb.classes_, index=df.index)
        df = pd.concat([df, genre_feature_df], axis=1)
        
        self.mlb = mlb
        return df, mlb
    
    def process_user_features(self, df: pd.DataFrame, users_df: pd.DataFrame) -> Tuple[pd.DataFrame, OneHotEncoder]:
        """Processes user features using OneHotEncoder"""
        ohe_user_features = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        ohe_user_features.fit(users_df[['Gender', 'Age', 'Occupation']])

        user_encoded_features = ohe_user_features.transform(df[['Gender', 'Age', 'Occupation']])
        user_feature_df = pd.DataFrame(user_encoded_features,
                                       columns=ohe_user_features.get_feature_names_out(['Gender', 'Age', 'Occupation']),
                                       index=df.index)
        df = pd.concat([df, user_feature_df], axis=1)
        
        self.ohe_user_features = ohe_user_features
        return df, ohe_user_features
    
    def prepare_features(self, df: pd.DataFrame, mlb: MultiLabelBinarizer, 
                        ohe_user_features: OneHotEncoder) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """Prepares final feature set for training"""
        numerical_features = ['ReleaseYear']
        categorical_features_genres = list(mlb.classes_)
        categorical_features_users = list(ohe_user_features.get_feature_names_out(['Gender', 'Age', 'Occupation']))

        feature_columns = numerical_features + categorical_features_genres + categorical_features_users
        target_column = 'Engagement'

        df_processed = df[feature_columns + [target_column]].dropna()

        X = df_processed[feature_columns]
        y = df_processed[target_column]

        print(f"Features for training: {feature_columns}")
        print(f"Preprocessed data shape (X, y): {X.shape}, {y.shape}")

        if X.empty or y.empty:
            raise ValueError("Preprocessed data is empty. Check data loading and cleaning steps.")
        if not np.issubdtype(y.dtype, np.integer):
            raise ValueError("Target variable 'Engagement' is not integer type after binarization.")

        self.feature_columns = feature_columns
        return X, y, feature_columns
    
    def fit_scaler(self, X_train: pd.DataFrame) -> StandardScaler:
        """Fits scaler on training data"""
        scaler = StandardScaler()
        scaler.fit(X_train[['ReleaseYear']])
        self.scaler = scaler
        return scaler
    
    def transform_data(self, X: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
        """Transforms data using fitted scaler"""
        X_transformed = X.copy()
        X_transformed['ReleaseYear'] = scaler.transform(X[['ReleaseYear']])
        return X_transformed
    
    def preprocess_data(self, ratings_df: pd.DataFrame, movies_df: pd.DataFrame, 
                       users_df: pd.DataFrame, base_path: str) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
        """Complete preprocessing pipeline"""
        # Create genres feature
        movies_df, genre_cols = self.create_genres_feature(movies_df)
        
        # Extract release year
        release_year_df = self.extract_release_year(movies_df, base_path)
        
        # Binarize ratings
        ratings_df = self.binarize_ratings(ratings_df)
        
        # Merge datasets
        df = self.merge_datasets(ratings_df, movies_df, users_df, release_year_df)
        
        # Process genres
        df, mlb = self.process_genres(df, genre_cols)
        
        # Process user features
        df, ohe_user_features = self.process_user_features(df, users_df)
        
        # Prepare final features
        X, y, feature_columns = self.prepare_features(df, mlb, ohe_user_features)
        
        preprocessing_artifacts = {
            'mlb': mlb,
            'ohe_user_features': ohe_user_features,
            'feature_columns': feature_columns
        }
        
        return X, y, preprocessing_artifacts 