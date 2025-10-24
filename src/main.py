"""
Main program for traffic flow prediction
Integrates the entire workflow from data loading to model evaluation
"""

import sys
import argparse
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent))

import config
from data_loader import DataLoader
from preprocess import DataPreprocessor
from feature_engineering import FeatureEngineer
from train_models import ModelTrainer
from evaluate import ModelEvaluator


def main(args):
    """
    Main workflow for traffic flow prediction
    
    Args:
        args: Command line arguments
    """
    print("=" * 80)
    print("Traffic Flow Prediction System")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Data directory: {config.RAW_DATA_DIR}")
    print(f"  Models directory: {config.MODELS_DIR}")
    print(f"  Results directory: {config.RESULTS_DIR}")
    print("=" * 80 + "\n")
    
    # Step 1: Load Data
    print("\n" + "=" * 80)
    print("STEP 1: Data Loading")
    print("=" * 80)
    
    loader = DataLoader()
    
    if args.sample_data:
        # Create sample data
        df = loader.create_sample_data(
            num_samples=args.num_samples,
            start_date=args.start_date,
            freq=args.freq
        )
        loader.save_data(df, "traffic_data.csv")
    else:
        # Load from CSV files
        df = loader.load_all_csv()
        if df is None:
            print("\nNo data found. Creating sample data...")
            df = loader.create_sample_data(num_samples=10000)
            loader.save_data(df, "traffic_data.csv")
    
    print(f"\nData loaded: {len(df)} rows, {len(df.columns)} columns")
    
    # Step 2: Preprocessing
    print("\n" + "=" * 80)
    print("STEP 2: Data Preprocessing")
    print("=" * 80)
    
    preprocessor = DataPreprocessor()
    df_clean = preprocessor.clean_data(
        df,
        handle_missing=args.handle_missing,
        remove_outliers=args.remove_outliers,
        normalize=args.normalize
    )
    
    # Save cleaned data
    loader.save_data(df_clean, "cleaned_traffic_data.csv")
    
    # Step 3: Feature Engineering
    print("\n" + "=" * 80)
    print("STEP 3: Feature Engineering")
    print("=" * 80)
    
    engineer = FeatureEngineer()
    df_features = engineer.create_all_features(
        df_clean,
        include_lag=args.include_lag,
        include_rolling=args.include_rolling
    )
    
    # Save featured data
    loader.save_data(df_features, "featured_traffic_data.csv")
    
    # Step 4: Split Data
    print("\n" + "=" * 80)
    print("STEP 4: Data Splitting")
    print("=" * 80)
    
    train_df, val_df, test_df = preprocessor.split_data(df_features)
    
    # Step 5: Train Models
    print("\n" + "=" * 80)
    print("STEP 5: Model Training")
    print("=" * 80)
    
    trainer = ModelTrainer()
    
    if not args.skip_training:
        models = trainer.train_all_models(train_df, val_df)
        
        # Save models
        if args.save_models:
            trainer.save_all_models()
    else:
        print("Skipping training, loading existing models...")
        models = {}
        for model_name in ['xgboost', 'random_forest', 'svm']:
            model = trainer.load_model(model_name)
            if model:
                models[model_name] = model
    
    if not models:
        print("No models available for evaluation")
        return
    
    # Step 6: Evaluate Models
    print("\n" + "=" * 80)
    print("STEP 6: Model Evaluation")
    print("=" * 80)
    
    evaluator = ModelEvaluator()
    evaluator.evaluate_all_models(models, test_df)
    
    # Step 7: Compare Models
    print("\n" + "=" * 80)
    print("STEP 7: Model Comparison")
    print("=" * 80)
    
    evaluator.compare_models(save_plot=True, save_results=True)
    
    # Generate comprehensive report
    evaluator.generate_report()
    
    # Print best model
    best_model = evaluator.get_best_model(metric='MAE')
    print(f"\n{'='*80}")
    print(f"Best Model (by MAE): {best_model}")
    print(f"{'='*80}")
    
    # Final summary
    print("\n" + "=" * 80)
    print("Workflow Complete!")
    print("=" * 80)
    print("\nGenerated files:")
    print(f"  - Data: {config.PROCESSED_DATA_DIR}")
    print(f"  - Models: {config.MODELS_DIR}")
    print(f"  - Results: {config.RESULTS_DIR}")
    print("\nCheck the results directory for:")
    print("  - model_comparison.png: Visual comparison of all models")
    print("  - evaluation_results.txt: Detailed metrics for each model")
    print("  - evaluation_report.txt: Comprehensive evaluation report")
    print("  - *_predictions.png: Individual model prediction plots")
    print("=" * 80 + "\n")


def parse_arguments():
    """
    Parse command line arguments
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='Traffic Flow Prediction System',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data options
    parser.add_argument('--sample-data', action='store_true',
                       help='Create sample data instead of loading from files')
    parser.add_argument('--num-samples', type=int, default=10000,
                       help='Number of samples for sample data')
    parser.add_argument('--start-date', type=str, default='2023-01-01',
                       help='Start date for sample data')
    parser.add_argument('--freq', type=str, default='15min',
                       help='Frequency for sample data')
    
    # Preprocessing options
    parser.add_argument('--handle-missing', action='store_true', default=True,
                       help='Handle missing values')
    parser.add_argument('--remove-outliers', action='store_true', default=True,
                       help='Remove outliers')
    parser.add_argument('--normalize', action='store_true', default=False,
                       help='Normalize features')
    
    # Feature engineering options
    parser.add_argument('--include-lag', action='store_true', default=False,
                       help='Include lag features')
    parser.add_argument('--include-rolling', action='store_true', default=False,
                       help='Include rolling statistics features')
    
    # Training options
    parser.add_argument('--skip-training', action='store_true', default=False,
                       help='Skip training and load existing models')
    parser.add_argument('--save-models', action='store_true', default=True,
                       help='Save trained models')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
