#!/usr/bin/env python
"""
Quick verification script to test the complete traffic flow prediction workflow
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def main():
    print("=" * 80)
    print("Traffic Flow Prediction - Quick Verification")
    print("=" * 80)
    
    # Test imports
    print("\n[1/7] Testing imports...")
    try:
        from data_loader import DataLoader
        from preprocess import DataPreprocessor
        from feature_engineering import FeatureEngineer
        from train_models import ModelTrainer
        from evaluate import ModelEvaluator
        import config
        print("✅ All modules imported successfully")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    # Test data loading
    print("\n[2/7] Testing data loading...")
    loader = DataLoader()
    df = loader.create_sample_data(num_samples=1000)
    print(f"✅ Created {len(df)} sample records")
    
    # Test preprocessing
    print("\n[3/7] Testing preprocessing...")
    preprocessor = DataPreprocessor()
    df_clean = preprocessor.clean_data(df, handle_missing=True, 
                                       remove_outliers=False, normalize=False)
    print(f"✅ Cleaned data: {len(df_clean)} records")
    
    # Test feature engineering
    print("\n[4/7] Testing feature engineering...")
    engineer = FeatureEngineer()
    df_features = engineer.create_all_features(df_clean)
    print(f"✅ Created {len(df_features.columns)} features")
    
    # Test data splitting
    print("\n[5/7] Testing data splitting...")
    train_df, val_df, test_df = preprocessor.split_data(df_features)
    print(f"✅ Split: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test")
    
    # Test model training
    print("\n[6/7] Testing model training...")
    trainer = ModelTrainer()
    models = trainer.train_all_models(train_df, val_df)
    print(f"✅ Trained {len(models)} models: {list(models.keys())}")
    
    # Test evaluation
    print("\n[7/7] Testing evaluation...")
    evaluator = ModelEvaluator()
    results = evaluator.evaluate_all_models(models, test_df)
    print(f"✅ Evaluated {len(results)} models")
    
    # Summary
    print("\n" + "=" * 80)
    print("VERIFICATION COMPLETE")
    print("=" * 80)
    print("\n✅ All components working correctly!")
    print(f"\nBest model: {evaluator.get_best_model('MAE')}")
    print("\nProject structure verified:")
    print(f"  - Data directory: {config.DATA_DIR}")
    print(f"  - Models directory: {config.MODELS_DIR}")
    print(f"  - Results directory: {config.RESULTS_DIR}")
    print("=" * 80)
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
