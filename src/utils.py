"""
Utility functions for traffic flow prediction project
Includes time conversion, plotting, and helper functions
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Tuple
import config

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


def parse_datetime(dt_string: str, format: str = None) -> datetime:
    """
    Parse datetime string to datetime object
    
    Args:
        dt_string: Datetime string
        format: Datetime format (auto-detect if None)
    
    Returns:
        Datetime object
    """
    if format:
        return datetime.strptime(dt_string, format)
    else:
        return pd.to_datetime(dt_string)


def format_datetime(dt: datetime, format: str = '%Y-%m-%d %H:%M:%S') -> str:
    """
    Format datetime object to string
    
    Args:
        dt: Datetime object
        format: Output format
    
    Returns:
        Formatted datetime string
    """
    return dt.strftime(format)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate evaluation metrics
    
    Args:
        y_true: True values
        y_pred: Predicted values
    
    Returns:
        Dictionary of metrics
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
    r2 = r2_score(y_true, y_pred)
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'R2': r2
    }


def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray, 
                     model_name: str, save_path: str = None, 
                     show_samples: int = 200):
    """
    Plot predicted vs actual values
    
    Args:
        y_true: True values
        y_pred: Predicted values
        model_name: Name of the model
        save_path: Path to save plot (None to not save)
        show_samples: Number of samples to show
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Time series plot
    samples = min(show_samples, len(y_true))
    x = np.arange(samples)
    ax1.plot(x, y_true[:samples], label='Actual', alpha=0.7, linewidth=2)
    ax1.plot(x, y_pred[:samples], label='Predicted', alpha=0.7, linewidth=2)
    ax1.set_xlabel('Sample Index')
    ax1.set_ylabel('Traffic Flow')
    ax1.set_title(f'{model_name} - Predictions vs Actual')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Scatter plot
    ax2.scatter(y_true, y_pred, alpha=0.5, s=10)
    
    # Add perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    ax2.set_xlabel('Actual Traffic Flow')
    ax2.set_ylabel('Predicted Traffic Flow')
    ax2.set_title(f'{model_name} - Scatter Plot')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.close()


def plot_comparison(results: Dict[str, Dict], save_path: str = None):
    """
    Plot comparison of multiple models
    
    Args:
        results: Dictionary with model names as keys and metrics as values
        save_path: Path to save plot
    """
    models = list(results.keys())
    metrics = ['MAE', 'RMSE', 'MAPE', 'R2']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics):
        values = [results[model][metric] for model in models]
        
        ax = axes[idx]
        bars = ax.bar(models, values, alpha=0.7, edgecolor='black')
        
        # Color bars based on performance (lower is better for MAE, RMSE, MAPE)
        if metric in ['MAE', 'RMSE', 'MAPE']:
            colors = plt.cm.RdYlGn_r(np.linspace(0.3, 0.9, len(values)))
        else:  # R2 - higher is better
            colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(values)))
        
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        ax.set_ylabel(metric)
        ax.set_title(f'Model Comparison - {metric}')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, v in enumerate(values):
            ax.text(i, v, f'{v:.2f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to {save_path}")
    
    plt.close()


def save_results(results: Dict[str, Dict], filename: str):
    """
    Save results to a text file
    
    Args:
        results: Dictionary with model names as keys and metrics as values
        filename: Output filename
    """
    filepath = config.RESULTS_DIR / filename
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("Traffic Flow Prediction - Model Comparison Results\n")
        f.write("=" * 60 + "\n\n")
        
        for model_name, metrics in results.items():
            f.write(f"\n{model_name}:\n")
            f.write("-" * 40 + "\n")
            for metric_name, value in metrics.items():
                f.write(f"  {metric_name:8s}: {value:10.4f}\n")
        
        f.write("\n" + "=" * 60 + "\n")
    
    print(f"Results saved to {filepath}")


def print_metrics(model_name: str, metrics: Dict[str, float]):
    """
    Print metrics in a formatted way
    
    Args:
        model_name: Name of the model
        metrics: Dictionary of metrics
    """
    print(f"\n{'='*50}")
    print(f"{model_name} Performance Metrics")
    print(f"{'='*50}")
    for metric, value in metrics.items():
        print(f"{metric:8s}: {value:10.4f}")
    print(f"{'='*50}\n")
