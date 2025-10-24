"""
Feature engineering for traffic flow prediction
Handles traffic-specific features like peak hours and holidays
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import List
import config


class FeatureEngineer:
    """
    Create features for traffic flow prediction
    """
    
    def __init__(self):
        """Initialize feature engineer"""
        pass
    
    def extract_time_features(self, df: pd.DataFrame, 
                             time_column: str = None) -> pd.DataFrame:
        """
        Extract time-based features from timestamp column
        
        Args:
            df: Input DataFrame
            time_column: Name of timestamp column
        
        Returns:
            DataFrame with time features added
        """
        time_column = time_column or config.TIME_COLUMN
        
        print(f"\nExtracting time features from '{time_column}'...")
        
        df_features = df.copy()
        
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(df_features[time_column]):
            df_features[time_column] = pd.to_datetime(df_features[time_column])
        
        # Extract basic time features
        df_features['hour'] = df_features[time_column].dt.hour
        df_features['day_of_week'] = df_features[time_column].dt.dayofweek  # 0=Monday, 6=Sunday
        df_features['day_of_month'] = df_features[time_column].dt.day
        df_features['month'] = df_features[time_column].dt.month
        df_features['day_of_year'] = df_features[time_column].dt.dayofyear
        df_features['week_of_year'] = df_features[time_column].dt.isocalendar().week
        
        # Cyclical encoding for hour (24-hour cycle)
        df_features['hour_sin'] = np.sin(2 * np.pi * df_features['hour'] / 24)
        df_features['hour_cos'] = np.cos(2 * np.pi * df_features['hour'] / 24)
        
        # Cyclical encoding for day of week (7-day cycle)
        df_features['day_sin'] = np.sin(2 * np.pi * df_features['day_of_week'] / 7)
        df_features['day_cos'] = np.cos(2 * np.pi * df_features['day_of_week'] / 7)
        
        # Cyclical encoding for month (12-month cycle)
        df_features['month_sin'] = np.sin(2 * np.pi * df_features['month'] / 12)
        df_features['month_cos'] = np.cos(2 * np.pi * df_features['month'] / 12)
        
        print(f"Added time features: hour, day_of_week, month, etc.")
        
        return df_features
    
    def add_weekend_feature(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add weekend indicator feature
        
        Args:
            df: Input DataFrame (must have 'day_of_week' column)
        
        Returns:
            DataFrame with weekend feature
        """
        print("\nAdding weekend feature...")
        
        df_features = df.copy()
        
        if 'day_of_week' not in df_features.columns:
            raise ValueError("DataFrame must have 'day_of_week' column. Run extract_time_features first.")
        
        # Weekend is Saturday (5) and Sunday (6)
        df_features['is_weekend'] = (df_features['day_of_week'] >= 5).astype(int)
        
        print(f"Weekend samples: {df_features['is_weekend'].sum()} ({df_features['is_weekend'].mean()*100:.1f}%)")
        
        return df_features
    
    def add_peak_hour_feature(self, df: pd.DataFrame,
                             morning_start: int = None,
                             morning_end: int = None,
                             evening_start: int = None,
                             evening_end: int = None) -> pd.DataFrame:
        """
        Add peak hour indicator feature
        
        Args:
            df: Input DataFrame (must have 'hour' column)
            morning_start: Morning peak start hour
            morning_end: Morning peak end hour
            evening_start: Evening peak start hour
            evening_end: Evening peak end hour
        
        Returns:
            DataFrame with peak hour feature
        """
        morning_start = morning_start or config.MORNING_PEAK_START
        morning_end = morning_end or config.MORNING_PEAK_END
        evening_start = evening_start or config.EVENING_PEAK_START
        evening_end = evening_end or config.EVENING_PEAK_END
        
        print(f"\nAdding peak hour feature...")
        print(f"Morning peak: {morning_start}:00 - {morning_end}:00")
        print(f"Evening peak: {evening_start}:00 - {evening_end}:00")
        
        df_features = df.copy()
        
        if 'hour' not in df_features.columns:
            raise ValueError("DataFrame must have 'hour' column. Run extract_time_features first.")
        
        # Check if hour is in morning or evening peak
        morning_peak = (df_features['hour'] >= morning_start) & (df_features['hour'] < morning_end)
        evening_peak = (df_features['hour'] >= evening_start) & (df_features['hour'] < evening_end)
        
        df_features['is_peak_hour'] = (morning_peak | evening_peak).astype(int)
        
        print(f"Peak hour samples: {df_features['is_peak_hour'].sum()} ({df_features['is_peak_hour'].mean()*100:.1f}%)")
        
        return df_features
    
    def add_holiday_feature(self, df: pd.DataFrame,
                           time_column: str = None,
                           holidays: List[str] = None) -> pd.DataFrame:
        """
        Add holiday indicator feature
        
        Args:
            df: Input DataFrame
            time_column: Name of timestamp column
            holidays: List of holidays in 'MM-DD' format
        
        Returns:
            DataFrame with holiday feature
        """
        time_column = time_column or config.TIME_COLUMN
        holidays = holidays or config.HOLIDAYS
        
        print(f"\nAdding holiday feature with {len(holidays)} holidays...")
        
        df_features = df.copy()
        
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(df_features[time_column]):
            df_features[time_column] = pd.to_datetime(df_features[time_column])
        
        # Create MM-DD string for comparison
        df_features['date_str'] = df_features[time_column].dt.strftime('%m-%d')
        
        # Check if date is in holiday list
        df_features['is_holiday'] = df_features['date_str'].isin(holidays).astype(int)
        
        # Drop temporary column
        df_features = df_features.drop(columns=['date_str'])
        
        print(f"Holiday samples: {df_features['is_holiday'].sum()} ({df_features['is_holiday'].mean()*100:.1f}%)")
        
        return df_features
    
    def add_lag_features(self, df: pd.DataFrame,
                        target_column: str = None,
                        lags: List[int] = [1, 2, 3, 6, 12, 24]) -> pd.DataFrame:
        """
        Add lag features (previous values)
        
        Args:
            df: Input DataFrame
            target_column: Target column to create lags from
            lags: List of lag periods
        
        Returns:
            DataFrame with lag features
        """
        target_column = target_column or config.TARGET_COLUMN
        
        print(f"\nAdding lag features for '{target_column}'...")
        print(f"Lag periods: {lags}")
        
        df_features = df.copy()
        
        for lag in lags:
            df_features[f'{target_column}_lag_{lag}'] = df_features[target_column].shift(lag)
        
        # Drop rows with NaN from lag features
        initial_len = len(df_features)
        df_features = df_features.dropna()
        print(f"Dropped {initial_len - len(df_features)} rows due to lag NaN values")
        
        return df_features
    
    def add_rolling_features(self, df: pd.DataFrame,
                            target_column: str = None,
                            windows: List[int] = [3, 6, 12, 24]) -> pd.DataFrame:
        """
        Add rolling statistics features
        
        Args:
            df: Input DataFrame
            target_column: Target column to calculate rolling stats
            windows: List of window sizes
        
        Returns:
            DataFrame with rolling features
        """
        target_column = target_column or config.TARGET_COLUMN
        
        print(f"\nAdding rolling features for '{target_column}'...")
        print(f"Window sizes: {windows}")
        
        df_features = df.copy()
        
        for window in windows:
            # Rolling mean
            df_features[f'{target_column}_rolling_mean_{window}'] = \
                df_features[target_column].rolling(window=window, min_periods=1).mean()
            
            # Rolling std
            df_features[f'{target_column}_rolling_std_{window}'] = \
                df_features[target_column].rolling(window=window, min_periods=1).std()
            
            # Rolling min/max
            df_features[f'{target_column}_rolling_min_{window}'] = \
                df_features[target_column].rolling(window=window, min_periods=1).min()
            df_features[f'{target_column}_rolling_max_{window}'] = \
                df_features[target_column].rolling(window=window, min_periods=1).max()
        
        return df_features
    
    def create_all_features(self, df: pd.DataFrame,
                           include_lag: bool = False,
                           include_rolling: bool = False) -> pd.DataFrame:
        """
        Create all traffic-related features
        
        Args:
            df: Input DataFrame
            include_lag: Whether to include lag features
            include_rolling: Whether to include rolling features
        
        Returns:
            DataFrame with all features
        """
        print("=" * 60)
        print("Creating All Traffic Features")
        print("=" * 60)
        
        df_features = df.copy()
        
        # Time features
        df_features = self.extract_time_features(df_features)
        
        # Weekend feature
        df_features = self.add_weekend_feature(df_features)
        
        # Peak hour feature
        df_features = self.add_peak_hour_feature(df_features)
        
        # Holiday feature
        df_features = self.add_holiday_feature(df_features)
        
        # Optional: Lag features
        if include_lag:
            df_features = self.add_lag_features(df_features)
        
        # Optional: Rolling features
        if include_rolling:
            df_features = self.add_rolling_features(df_features)
        
        print("\n" + "=" * 60)
        print("Feature Engineering Complete")
        print("=" * 60)
        print(f"Total features: {len(df_features.columns)}")
        print(f"Feature columns: {list(df_features.columns)}")
        
        return df_features


def main():
    """
    Main function for testing feature engineering
    """
    print("=" * 60)
    print("Traffic Flow Feature Engineering")
    print("=" * 60)
    
    # Load sample data
    from data_loader import DataLoader
    loader = DataLoader()
    df = loader.create_sample_data(num_samples=1000)
    
    print("\n" + "=" * 60)
    print("Original data:")
    print("=" * 60)
    print(df.head())
    print(f"Shape: {df.shape}")
    
    # Create features
    engineer = FeatureEngineer()
    df_features = engineer.create_all_features(df, include_lag=False, include_rolling=False)
    
    print("\n" + "=" * 60)
    print("Data with features:")
    print("=" * 60)
    print(df_features.head())
    print(f"Shape: {df_features.shape}")
    
    # Show feature statistics
    print("\n" + "=" * 60)
    print("Feature Statistics:")
    print("=" * 60)
    print(f"Weekend ratio: {df_features['is_weekend'].mean():.2%}")
    print(f"Peak hour ratio: {df_features['is_peak_hour'].mean():.2%}")
    print(f"Holiday ratio: {df_features['is_holiday'].mean():.2%}")


if __name__ == "__main__":
    main()
