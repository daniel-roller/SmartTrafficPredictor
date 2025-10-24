"""
Data loader for traffic flow prediction project
Reads and integrates CSV files from raw data directory
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional
import config


class DataLoader:
    """
    Load and integrate traffic flow data from CSV files
    """
    
    def __init__(self, data_dir: Path = None):
        """
        Initialize data loader
        
        Args:
            data_dir: Directory containing raw data files
        """
        self.data_dir = data_dir or config.RAW_DATA_DIR
        self.data = None
    
    def load_csv(self, filepath: Path) -> pd.DataFrame:
        """
        Load a single CSV file
        
        Args:
            filepath: Path to CSV file
        
        Returns:
            DataFrame with loaded data
        """
        try:
            df = pd.read_csv(filepath, encoding='utf-8')
            print(f"Loaded {len(df)} rows from {filepath.name}")
            return df
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None
    
    def load_all_csv(self, pattern: str = "*.csv") -> pd.DataFrame:
        """
        Load all CSV files matching pattern from data directory
        
        Args:
            pattern: File pattern to match (default: *.csv)
        
        Returns:
            Combined DataFrame
        """
        csv_files = list(self.data_dir.glob(pattern))
        
        if not csv_files:
            print(f"No CSV files found in {self.data_dir}")
            return None
        
        print(f"Found {len(csv_files)} CSV files")
        
        dfs = []
        for filepath in csv_files:
            df = self.load_csv(filepath)
            if df is not None:
                dfs.append(df)
        
        if not dfs:
            print("No data loaded successfully")
            return None
        
        # Combine all DataFrames
        combined_df = pd.concat(dfs, ignore_index=True)
        print(f"\nTotal rows after combining: {len(combined_df)}")
        
        self.data = combined_df
        return combined_df
    
    def create_sample_data(self, 
                          num_samples: int = 10000,
                          start_date: str = '2023-01-01',
                          freq: str = '15min') -> pd.DataFrame:
        """
        Create sample traffic flow data for testing
        
        Args:
            num_samples: Number of samples to generate
            start_date: Start date for time series
            freq: Frequency of data points
        
        Returns:
            DataFrame with sample data
        """
        print(f"Creating sample data with {num_samples} samples...")
        
        # Generate datetime index
        date_range = pd.date_range(start=start_date, periods=num_samples, freq=freq)
        
        # Extract time features
        hours = date_range.hour
        days_of_week = date_range.dayofweek
        
        # Base traffic flow with daily and weekly patterns
        base_flow = 1000
        
        # Hourly pattern (higher during peak hours)
        hourly_pattern = np.sin((hours - 6) * np.pi / 12) * 300 + 200
        
        # Weekly pattern (lower on weekends)
        weekly_pattern = np.where(days_of_week < 5, 200, -200)
        
        # Add noise
        noise = np.random.normal(0, 100, num_samples)
        
        # Combine patterns
        traffic_flow = base_flow + hourly_pattern + weekly_pattern + noise
        traffic_flow = np.maximum(traffic_flow, 100)  # Ensure positive values
        
        # Create DataFrame
        df = pd.DataFrame({
            'timestamp': date_range,
            'traffic_flow': traffic_flow.astype(int)
        })
        
        print(f"Sample data created: {len(df)} rows")
        print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        
        self.data = df
        return df
    
    def save_data(self, df: pd.DataFrame, filename: str):
        """
        Save DataFrame to CSV
        
        Args:
            df: DataFrame to save
            filename: Output filename
        """
        filepath = config.PROCESSED_DATA_DIR / filename
        df.to_csv(filepath, index=False)
        print(f"Data saved to {filepath}")
    
    def get_data_info(self) -> dict:
        """
        Get information about loaded data
        
        Returns:
            Dictionary with data statistics
        """
        if self.data is None:
            return {"error": "No data loaded"}
        
        info = {
            "num_rows": len(self.data),
            "num_columns": len(self.data.columns),
            "columns": list(self.data.columns),
            "missing_values": self.data.isnull().sum().to_dict(),
            "memory_usage": f"{self.data.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
        }
        
        return info


def main():
    """
    Main function for testing data loader
    """
    print("=" * 60)
    print("Traffic Flow Data Loader")
    print("=" * 60)
    
    loader = DataLoader()
    
    # Try to load existing CSV files
    df = loader.load_all_csv()
    
    # If no data found, create sample data
    if df is None:
        print("\nNo CSV files found. Creating sample data...")
        df = loader.create_sample_data(num_samples=10000)
        
        # Save sample data
        loader.save_data(df, "sample_traffic_data.csv")
    
    # Print data info
    print("\n" + "=" * 60)
    print("Data Information:")
    print("=" * 60)
    info = loader.get_data_info()
    for key, value in info.items():
        print(f"{key}: {value}")
    
    # Print first few rows
    print("\n" + "=" * 60)
    print("First 5 rows:")
    print("=" * 60)
    print(df.head())
    
    print("\n" + "=" * 60)
    print("Data statistics:")
    print("=" * 60)
    print(df.describe())


if __name__ == "__main__":
    main()
