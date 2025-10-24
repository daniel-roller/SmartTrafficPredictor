"""
Data preprocessing for traffic flow prediction
Handles data cleaning and standardization
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Tuple, Optional
import config


class DataPreprocessor:
    """
    Preprocess traffic flow data
    """
    
    def __init__(self):
        """Initialize preprocessor"""
        self.scaler = None
        self.feature_columns = None
    
    def handle_missing_values(self, df: pd.DataFrame, method: str = 'interpolate') -> pd.DataFrame:
        """
        Handle missing values in the dataset
        
        Args:
            df: Input DataFrame
            method: Method to handle missing values ('drop', 'fill', 'interpolate')
        
        Returns:
            DataFrame with missing values handled
        """
        print(f"\nHandling missing values using method: {method}")
        print(f"Missing values before: {df.isnull().sum().sum()}")
        
        df_clean = df.copy()
        
        if method == 'drop':
            df_clean = df_clean.dropna()
        elif method == 'fill':
            # Fill with mean for numeric columns
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
            df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].mean())
        elif method == 'interpolate':
            # Use time-based interpolation
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
            df_clean[numeric_cols] = df_clean[numeric_cols].interpolate(method='time')
        
        print(f"Missing values after: {df_clean.isnull().sum().sum()}")
        
        return df_clean
    
    def remove_outliers(self, df: pd.DataFrame, columns: list = None, 
                       method: str = 'iqr', threshold: float = 3.0) -> pd.DataFrame:
        """
        Remove outliers from specified columns
        
        Args:
            df: Input DataFrame
            columns: List of columns to check for outliers (None for all numeric)
            method: Method to detect outliers ('iqr' or 'zscore')
            threshold: Threshold for outlier detection
        
        Returns:
            DataFrame with outliers removed
        """
        print(f"\nRemoving outliers using method: {method}")
        print(f"Rows before: {len(df)}")
        
        df_clean = df.copy()
        
        if columns is None:
            columns = df_clean.select_dtypes(include=[np.number]).columns.tolist()
            if config.TIME_COLUMN in columns:
                columns.remove(config.TIME_COLUMN)
        
        for col in columns:
            if col not in df_clean.columns:
                continue
            
            if method == 'iqr':
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                mask = (df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)
                df_clean = df_clean[mask]
                
            elif method == 'zscore':
                z_scores = np.abs((df_clean[col] - df_clean[col].mean()) / df_clean[col].std())
                df_clean = df_clean[z_scores < threshold]
        
        print(f"Rows after: {len(df_clean)}")
        print(f"Removed {len(df) - len(df_clean)} outliers")
        
        return df_clean
    
    def normalize_data(self, df: pd.DataFrame, columns: list = None,
                      method: str = 'standard') -> pd.DataFrame:
        """
        Normalize/standardize data
        
        Args:
            df: Input DataFrame
            columns: Columns to normalize (None for all numeric except time)
            method: Normalization method ('standard' or 'minmax')
        
        Returns:
            DataFrame with normalized data
        """
        print(f"\nNormalizing data using method: {method}")
        
        df_normalized = df.copy()
        
        if columns is None:
            columns = df_normalized.select_dtypes(include=[np.number]).columns.tolist()
            # Don't normalize time column or target if present
            exclude_cols = [config.TIME_COLUMN]
            columns = [col for col in columns if col not in exclude_cols]
        
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        if columns:
            df_normalized[columns] = self.scaler.fit_transform(df_normalized[columns])
            print(f"Normalized {len(columns)} columns")
        
        self.feature_columns = columns
        
        return df_normalized
    
    def clean_data(self, df: pd.DataFrame, 
                   handle_missing: bool = True,
                   remove_outliers: bool = True,
                   normalize: bool = False) -> pd.DataFrame:
        """
        Complete data cleaning pipeline
        
        Args:
            df: Input DataFrame
            handle_missing: Whether to handle missing values
            remove_outliers: Whether to remove outliers
            normalize: Whether to normalize data
        
        Returns:
            Cleaned DataFrame
        """
        print("=" * 60)
        print("Starting Data Cleaning Pipeline")
        print("=" * 60)
        
        df_clean = df.copy()
        
        # Handle missing values
        if handle_missing:
            df_clean = self.handle_missing_values(df_clean)
        
        # Remove outliers
        if remove_outliers:
            df_clean = self.remove_outliers(df_clean, columns=[config.TARGET_COLUMN])
        
        # Normalize data
        if normalize:
            df_clean = self.normalize_data(df_clean)
        
        print("\n" + "=" * 60)
        print("Data Cleaning Complete")
        print("=" * 60)
        print(f"Final shape: {df_clean.shape}")
        
        return df_clean
    
    def split_data(self, df: pd.DataFrame, 
                   train_ratio: float = None,
                   val_ratio: float = None,
                   test_ratio: float = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train, validation, and test sets
        
        Args:
            df: Input DataFrame
            train_ratio: Ratio for training set
            val_ratio: Ratio for validation set
            test_ratio: Ratio for test set
        
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        train_ratio = train_ratio or config.TRAIN_RATIO
        val_ratio = val_ratio or config.VAL_RATIO
        test_ratio = test_ratio or config.TEST_RATIO
        
        # Ensure ratios sum to 1
        total = train_ratio + val_ratio + test_ratio
        train_ratio /= total
        val_ratio /= total
        test_ratio /= total
        
        n = len(df)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        
        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[train_end:val_end].copy()
        test_df = df.iloc[val_end:].copy()
        
        print("\n" + "=" * 60)
        print("Data Split:")
        print("=" * 60)
        print(f"Train set: {len(train_df)} samples ({len(train_df)/n*100:.1f}%)")
        print(f"Val set:   {len(val_df)} samples ({len(val_df)/n*100:.1f}%)")
        print(f"Test set:  {len(test_df)} samples ({len(test_df)/n*100:.1f}%)")
        
        return train_df, val_df, test_df


def main():
    """
    Main function for testing preprocessor
    """
    print("=" * 60)
    print("Traffic Flow Data Preprocessor")
    print("=" * 60)
    
    # Load sample data
    from data_loader import DataLoader
    loader = DataLoader()
    df = loader.create_sample_data(num_samples=1000)
    
    # Add some missing values and outliers for testing
    df.loc[10:15, 'traffic_flow'] = np.nan
    df.loc[50:52, 'traffic_flow'] = 10000  # Outliers
    
    print("\n" + "=" * 60)
    print("Original data info:")
    print("=" * 60)
    print(df.describe())
    
    # Preprocess data
    preprocessor = DataPreprocessor()
    df_clean = preprocessor.clean_data(
        df, 
        handle_missing=True,
        remove_outliers=True,
        normalize=False
    )
    
    print("\n" + "=" * 60)
    print("Cleaned data info:")
    print("=" * 60)
    print(df_clean.describe())
    
    # Split data
    train_df, val_df, test_df = preprocessor.split_data(df_clean)


if __name__ == "__main__":
    main()
