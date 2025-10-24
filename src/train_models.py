"""
Model training for traffic flow prediction
Trains XGBoost, Random Forest, and SVM models
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, Tuple
import config

# Model imports
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available. Install with: pip install xgboost")


class ModelTrainer:
    """
    Train multiple models for traffic flow prediction
    """
    
    def __init__(self):
        """Initialize model trainer"""
        self.models = {}
        self.feature_columns = None
    
    def prepare_data(self, df: pd.DataFrame, 
                     target_column: str = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and target for training
        
        Args:
            df: Input DataFrame with features
            target_column: Name of target column
        
        Returns:
            Tuple of (X, y)
        """
        target_column = target_column or config.TARGET_COLUMN
        
        # Separate features and target
        X = df.drop(columns=[target_column, config.TIME_COLUMN], errors='ignore')
        y = df[target_column]
        
        # Store feature columns
        self.feature_columns = X.columns.tolist()
        
        print(f"\nPrepared data:")
        print(f"Features shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        print(f"Feature columns: {self.feature_columns}")
        
        return X, y
    
    def train_xgboost(self, X_train: pd.DataFrame, y_train: pd.Series,
                     X_val: pd.DataFrame = None, y_val: pd.Series = None,
                     params: dict = None) -> object:
        """
        Train XGBoost model
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            params: Model parameters
        
        Returns:
            Trained model
        """
        if not XGBOOST_AVAILABLE:
            print("XGBoost not available, skipping...")
            return None
        
        print("\n" + "=" * 60)
        print("Training XGBoost Model")
        print("=" * 60)
        
        params = params or config.MODEL_PARAMS['xgboost']
        
        model = xgb.XGBRegressor(**params)
        
        # Train with validation set if provided
        if X_val is not None and y_val is not None:
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
        else:
            model.fit(X_train, y_train)
        
        self.models['xgboost'] = model
        print("XGBoost training complete")
        
        return model
    
    def train_random_forest(self, X_train: pd.DataFrame, y_train: pd.Series,
                           params: dict = None) -> object:
        """
        Train Random Forest model
        
        Args:
            X_train: Training features
            y_train: Training target
            params: Model parameters
        
        Returns:
            Trained model
        """
        print("\n" + "=" * 60)
        print("Training Random Forest Model")
        print("=" * 60)
        
        params = params or config.MODEL_PARAMS['random_forest']
        
        model = RandomForestRegressor(**params)
        model.fit(X_train, y_train)
        
        self.models['random_forest'] = model
        print("Random Forest training complete")
        
        return model
    
    def train_svm(self, X_train: pd.DataFrame, y_train: pd.Series,
                  params: dict = None) -> object:
        """
        Train SVM model
        
        Args:
            X_train: Training features
            y_train: Training target
            params: Model parameters
        
        Returns:
            Trained model
        """
        print("\n" + "=" * 60)
        print("Training SVM Model")
        print("=" * 60)
        
        params = params or config.MODEL_PARAMS['svm']
        
        # For large datasets, use a subset for SVM training
        max_samples = 5000
        if len(X_train) > max_samples:
            print(f"Using {max_samples} samples for SVM training (dataset is large)")
            indices = np.random.choice(len(X_train), max_samples, replace=False)
            X_train_subset = X_train.iloc[indices]
            y_train_subset = y_train.iloc[indices]
        else:
            X_train_subset = X_train
            y_train_subset = y_train
        
        model = SVR(**params)
        model.fit(X_train_subset, y_train_subset)
        
        self.models['svm'] = model
        print("SVM training complete")
        
        return model
    
    def train_all_models(self, train_df: pd.DataFrame, 
                        val_df: pd.DataFrame = None,
                        target_column: str = None) -> Dict[str, object]:
        """
        Train all models
        
        Args:
            train_df: Training DataFrame
            val_df: Validation DataFrame
            target_column: Name of target column
        
        Returns:
            Dictionary of trained models
        """
        print("=" * 60)
        print("Training All Models")
        print("=" * 60)
        
        # Prepare training data
        X_train, y_train = self.prepare_data(train_df, target_column)
        
        # Prepare validation data if provided
        if val_df is not None:
            X_val, y_val = self.prepare_data(val_df, target_column)
        else:
            X_val, y_val = None, None
        
        # Train XGBoost
        if XGBOOST_AVAILABLE:
            self.train_xgboost(X_train, y_train, X_val, y_val)
        
        # Train Random Forest
        self.train_random_forest(X_train, y_train)
        
        # Train SVM
        self.train_svm(X_train, y_train)
        
        print("\n" + "=" * 60)
        print("All Models Trained")
        print("=" * 60)
        print(f"Trained models: {list(self.models.keys())}")
        
        return self.models
    
    def save_model(self, model_name: str, filename: str = None):
        """
        Save a trained model
        
        Args:
            model_name: Name of the model to save
            filename: Output filename (None for auto-generated)
        """
        if model_name not in self.models:
            print(f"Model '{model_name}' not found")
            return
        
        filename = filename or f"{model_name}_model.pkl"
        filepath = config.MODELS_DIR / filename
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.models[model_name], f)
        
        print(f"Model '{model_name}' saved to {filepath}")
    
    def save_all_models(self):
        """Save all trained models"""
        print("\n" + "=" * 60)
        print("Saving All Models")
        print("=" * 60)
        
        for model_name in self.models.keys():
            self.save_model(model_name)
        
        # Save feature columns
        feature_file = config.MODELS_DIR / "feature_columns.pkl"
        with open(feature_file, 'wb') as f:
            pickle.dump(self.feature_columns, f)
        print(f"Feature columns saved to {feature_file}")
    
    def load_model(self, model_name: str, filename: str = None) -> object:
        """
        Load a trained model
        
        Args:
            model_name: Name of the model
            filename: Model filename (None for auto-generated)
        
        Returns:
            Loaded model
        """
        filename = filename or f"{model_name}_model.pkl"
        filepath = config.MODELS_DIR / filename
        
        if not filepath.exists():
            print(f"Model file not found: {filepath}")
            return None
        
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        
        self.models[model_name] = model
        print(f"Model '{model_name}' loaded from {filepath}")
        
        return model
    
    def predict(self, model_name: str, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions with a trained model
        
        Args:
            model_name: Name of the model
            X: Features for prediction
        
        Returns:
            Predictions
        """
        if model_name not in self.models:
            print(f"Model '{model_name}' not found")
            return None
        
        return self.models[model_name].predict(X)


def main():
    """
    Main function for testing model training
    """
    print("=" * 60)
    print("Traffic Flow Model Training")
    print("=" * 60)
    
    # Load and prepare data
    from data_loader import DataLoader
    from preprocess import DataPreprocessor
    from feature_engineering import FeatureEngineer
    
    # Create sample data
    loader = DataLoader()
    df = loader.create_sample_data(num_samples=5000)
    
    # Preprocess
    preprocessor = DataPreprocessor()
    df_clean = preprocessor.clean_data(df, handle_missing=True, 
                                       remove_outliers=False, normalize=False)
    
    # Feature engineering
    engineer = FeatureEngineer()
    df_features = engineer.create_all_features(df_clean)
    
    # Split data
    train_df, val_df, test_df = preprocessor.split_data(df_features)
    
    # Train models
    trainer = ModelTrainer()
    models = trainer.train_all_models(train_df, val_df)
    
    # Save models
    trainer.save_all_models()
    
    print("\n" + "=" * 60)
    print("Model Training Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
