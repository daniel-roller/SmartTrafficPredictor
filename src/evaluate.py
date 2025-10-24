"""
Model evaluation for traffic flow prediction
Assesses and compares different model results
"""

import pandas as pd
import numpy as np
from typing import Dict, List
import config
import utils


class ModelEvaluator:
    """
    Evaluate and compare traffic flow prediction models
    """
    
    def __init__(self):
        """Initialize evaluator"""
        self.results = {}
    
    def evaluate_model(self, model_name: str, y_true: np.ndarray, 
                      y_pred: np.ndarray, save_plot: bool = True) -> Dict[str, float]:
        """
        Evaluate a single model
        
        Args:
            model_name: Name of the model
            y_true: True values
            y_pred: Predicted values
            save_plot: Whether to save prediction plot
        
        Returns:
            Dictionary of metrics
        """
        print(f"\n{'='*60}")
        print(f"Evaluating {model_name}")
        print(f"{'='*60}")
        
        # Calculate metrics
        metrics = utils.calculate_metrics(y_true, y_pred)
        
        # Store results
        self.results[model_name] = metrics
        
        # Print metrics
        utils.print_metrics(model_name, metrics)
        
        # Plot predictions
        if save_plot:
            plot_path = config.RESULTS_DIR / f"{model_name}_predictions.png"
            utils.plot_predictions(y_true, y_pred, model_name, str(plot_path))
        
        return metrics
    
    def evaluate_all_models(self, models: Dict, test_df: pd.DataFrame,
                          target_column: str = None) -> Dict[str, Dict[str, float]]:
        """
        Evaluate all models on test data
        
        Args:
            models: Dictionary of trained models
            test_df: Test DataFrame
            target_column: Name of target column
        
        Returns:
            Dictionary of results for all models
        """
        target_column = target_column or config.TARGET_COLUMN
        
        print("=" * 60)
        print("Evaluating All Models")
        print("=" * 60)
        
        # Prepare test data
        X_test = test_df.drop(columns=[target_column, config.TIME_COLUMN], errors='ignore')
        y_test = test_df[target_column].values
        
        print(f"\nTest set size: {len(y_test)}")
        
        # Evaluate each model
        for model_name, model in models.items():
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Evaluate
            self.evaluate_model(model_name, y_test, y_pred, save_plot=True)
        
        return self.results
    
    def compare_models(self, save_plot: bool = True, save_results: bool = True):
        """
        Compare all evaluated models
        
        Args:
            save_plot: Whether to save comparison plot
            save_results: Whether to save results to file
        """
        if not self.results:
            print("No results to compare. Run evaluate_all_models first.")
            return
        
        print("\n" + "=" * 60)
        print("Model Comparison")
        print("=" * 60)
        
        # Print comparison table
        print("\n{:<20} {:>10} {:>10} {:>10} {:>10}".format(
            "Model", "MAE", "RMSE", "MAPE", "R2"))
        print("-" * 65)
        
        for model_name, metrics in self.results.items():
            print("{:<20} {:>10.2f} {:>10.2f} {:>10.2f} {:>10.4f}".format(
                model_name,
                metrics['MAE'],
                metrics['RMSE'],
                metrics['MAPE'],
                metrics['R2']
            ))
        
        # Find best model for each metric
        print("\n" + "=" * 60)
        print("Best Models by Metric:")
        print("=" * 60)
        
        best_mae = min(self.results.items(), key=lambda x: x[1]['MAE'])
        best_rmse = min(self.results.items(), key=lambda x: x[1]['RMSE'])
        best_mape = min(self.results.items(), key=lambda x: x[1]['MAPE'])
        best_r2 = max(self.results.items(), key=lambda x: x[1]['R2'])
        
        print(f"MAE:  {best_mae[0]} ({best_mae[1]['MAE']:.2f})")
        print(f"RMSE: {best_rmse[0]} ({best_rmse[1]['RMSE']:.2f})")
        print(f"MAPE: {best_mape[0]} ({best_mape[1]['MAPE']:.2f}%)")
        print(f"R2:   {best_r2[0]} ({best_r2[1]['R2']:.4f})")
        
        # Plot comparison
        if save_plot:
            plot_path = config.RESULTS_DIR / "model_comparison.png"
            utils.plot_comparison(self.results, str(plot_path))
        
        # Save results
        if save_results:
            utils.save_results(self.results, "evaluation_results.txt")
    
    def get_best_model(self, metric: str = 'MAE') -> str:
        """
        Get the best model based on a specific metric
        
        Args:
            metric: Metric to use for comparison
        
        Returns:
            Name of the best model
        """
        if not self.results:
            print("No results available")
            return None
        
        if metric not in ['MAE', 'RMSE', 'MAPE', 'R2']:
            print(f"Invalid metric: {metric}")
            return None
        
        if metric == 'R2':
            # Higher is better for R2
            best_model = max(self.results.items(), key=lambda x: x[1][metric])
        else:
            # Lower is better for MAE, RMSE, MAPE
            best_model = min(self.results.items(), key=lambda x: x[1][metric])
        
        return best_model[0]
    
    def generate_report(self, output_file: str = None):
        """
        Generate a comprehensive evaluation report
        
        Args:
            output_file: Output filename (None for default)
        """
        if not self.results:
            print("No results to report. Run evaluate_all_models first.")
            return
        
        output_file = output_file or "evaluation_report.txt"
        filepath = config.RESULTS_DIR / output_file
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("Traffic Flow Prediction - Comprehensive Evaluation Report\n")
            f.write("=" * 80 + "\n\n")
            
            # Summary statistics
            f.write("SUMMARY\n")
            f.write("-" * 80 + "\n")
            f.write(f"Number of models evaluated: {len(self.results)}\n")
            f.write(f"Models: {', '.join(self.results.keys())}\n\n")
            
            # Detailed results for each model
            f.write("DETAILED RESULTS\n")
            f.write("-" * 80 + "\n\n")
            
            for model_name, metrics in self.results.items():
                f.write(f"{model_name}:\n")
                f.write("  " + "-" * 40 + "\n")
                for metric_name, value in metrics.items():
                    f.write(f"  {metric_name:8s}: {value:12.4f}\n")
                f.write("\n")
            
            # Best models
            f.write("BEST MODELS BY METRIC\n")
            f.write("-" * 80 + "\n")
            
            best_mae = min(self.results.items(), key=lambda x: x[1]['MAE'])
            best_rmse = min(self.results.items(), key=lambda x: x[1]['RMSE'])
            best_mape = min(self.results.items(), key=lambda x: x[1]['MAPE'])
            best_r2 = max(self.results.items(), key=lambda x: x[1]['R2'])
            
            f.write(f"MAE:  {best_mae[0]:15s} ({best_mae[1]['MAE']:.4f})\n")
            f.write(f"RMSE: {best_rmse[0]:15s} ({best_rmse[1]['RMSE']:.4f})\n")
            f.write(f"MAPE: {best_mape[0]:15s} ({best_mape[1]['MAPE']:.4f}%)\n")
            f.write(f"R2:   {best_r2[0]:15s} ({best_r2[1]['R2']:.4f})\n\n")
            
            # Recommendations
            f.write("RECOMMENDATIONS\n")
            f.write("-" * 80 + "\n")
            
            # Overall best model (based on multiple metrics)
            avg_ranks = {}
            for model_name in self.results.keys():
                ranks = []
                
                # Rank for MAE (lower is better)
                mae_rank = sorted(self.results.keys(), 
                                key=lambda x: self.results[x]['MAE']).index(model_name)
                ranks.append(mae_rank)
                
                # Rank for RMSE (lower is better)
                rmse_rank = sorted(self.results.keys(), 
                                 key=lambda x: self.results[x]['RMSE']).index(model_name)
                ranks.append(rmse_rank)
                
                # Rank for R2 (higher is better)
                r2_rank = sorted(self.results.keys(), 
                               key=lambda x: self.results[x]['R2'], reverse=True).index(model_name)
                ranks.append(r2_rank)
                
                avg_ranks[model_name] = sum(ranks) / len(ranks)
            
            overall_best = min(avg_ranks.items(), key=lambda x: x[1])
            
            f.write(f"Overall best model: {overall_best[0]}\n")
            f.write(f"  (Average rank: {overall_best[1]:.2f})\n\n")
            
            f.write("Use this model for production deployment based on balanced performance\n")
            f.write("across multiple evaluation metrics.\n\n")
            
            f.write("=" * 80 + "\n")
        
        print(f"\nComprehensive report saved to {filepath}")


def main():
    """
    Main function for testing evaluation
    """
    print("=" * 60)
    print("Traffic Flow Model Evaluation")
    print("=" * 60)
    
    # Load and prepare data
    from data_loader import DataLoader
    from preprocess import DataPreprocessor
    from feature_engineering import FeatureEngineer
    from train_models import ModelTrainer
    
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
    
    # Evaluate models
    evaluator = ModelEvaluator()
    evaluator.evaluate_all_models(models, test_df)
    
    # Compare models
    evaluator.compare_models(save_plot=True, save_results=True)
    
    # Generate report
    evaluator.generate_report()
    
    print("\n" + "=" * 60)
    print("Evaluation Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
