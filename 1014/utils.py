# -*- coding: utf-8 -*-
"""
共用工具函數 (增強版)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import joblib
import os
from typing import Dict, List, Tuple, Any

# 修改 is_holiday 函數，加入快取機制
_holiday_cache = {}

def is_holiday(date):
    """判斷是否為假日（帶快取）"""
    date_str = date.strftime('%Y-%m-%d')
    
    if date_str in _holiday_cache:
        return _holiday_cache[date_str]
    
    # 簡化版本：只判斷週末和固定假日
    is_weekend = date.weekday() >= 5
    
    # 簡單的固定假日列表
    fixed_holidays = [
        '2021-01-01', '2021-12-25', '2022-01-01', '2022-12-25',
        '2023-01-01', '2023-12-25', '2024-01-01', '2024-12-25',
        '2025-01-01', '2025-12-25'
    ]
    
    is_fixed_holiday = date_str in fixed_holidays
    result = is_weekend or is_fixed_holiday
    
    _holiday_cache[date_str] = result
    return result

def is_peak_hour(hour):
    """判斷是否為尖峰時段"""
    return any(start <= hour < end for start, end in [(7, 9), (17, 19)])

def calculate_metrics(y_true, y_pred):
    """計算評估指標 (增強版)"""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    # 避免除零錯誤的 MAPE 計算
    y_true_safe = np.where(y_true == 0, 1e-8, y_true)
    mape = np.mean(np.abs((y_true - y_pred) / y_true_safe)) * 100
    
    # 新增其他指標
    max_error = np.max(np.abs(y_true - y_pred))
    mean_error = np.mean(y_true - y_pred)
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'MAPE': mape,
        'Max_Error': max_error,
        'Mean_Error': mean_error
    }

def plot_model_comparison(results_dict, save_path=None):
    """繪製模型比較圖 (增強版)"""
    models = list(results_dict.keys())
    metrics = ['MAE', 'RMSE', 'R2', 'MAPE']
    
    # 設定中文字體
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.ravel()
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, metric in enumerate(metrics):
        values = [results_dict[model][metric] for model in models]
        bars = axes[i].bar(models, values, color=colors[:len(models)])
        axes[i].set_title(f'{metric} 比較', fontsize=14, fontweight='bold')
        axes[i].set_ylabel(metric, fontsize=12)
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].grid(True, alpha=0.3)
        
        # 在柱狀圖上標註數值
        for bar, value in zip(bars, values):
            axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + bar.get_height()*0.01,
                       f'{value:.4f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 模型比較圖已儲存: {save_path}")
    
    if not save_path:  # 只有在沒有指定保存路徑時才顯示
        plt.show()
    else:
        plt.close()

def save_model(model, filepath):
    """儲存模型 (增強版)"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    # 不要每次都印出儲存訊息，以免輸出過多

def load_model(filepath):
    """載入模型"""
    return joblib.load(filepath)

def save_results(results, filepath):
    """儲存結果 (增強版)"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    if isinstance(results, dict):
        pd.DataFrame(results).to_csv(filepath, index=False, encoding='utf-8-sig')
    else:
        results.to_csv(filepath, index=False, encoding='utf-8-sig')
    # 不要每次都印出儲存訊息，以免輸出過多

def generate_model_summary_table(results_dict):
    """生成模型摘要表格"""
    summary_data = []
    for dataset_name, models_results in results_dict.items():
        for model_name, metrics in models_results.items():
            summary_data.append({
                'Dataset': dataset_name,
                'Model': model_name,
                'R²': f"{metrics['R2']:.4f}",
                'RMSE': f"{metrics['RMSE']:.2f}",
                'MAE': f"{metrics['MAE']:.2f}",
                'MAPE': f"{metrics['MAPE']:.2f}%"
            })
    
    return pd.DataFrame(summary_data)