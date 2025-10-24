# -*- coding: utf-8 -*-
"""
交通流量預測系統 - 工具函數
"""

import numpy as np
import pandas as pd
import pickle
import json
import os
import datetime
from typing import Dict, List, Tuple, Any, Optional, Union  # 增加 Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from config import config

def set_random_seed(seed: int = 42):
    """設定隨機種子確保結果可重現"""
    np.random.seed(seed)
    # 如果有其他套件也設定種子
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass

def create_scaler(method: str = 'standard'):
    """建立資料正規化器"""
    if method == 'standard':
        return StandardScaler()
    elif method == 'minmax':
        return MinMaxScaler()
    elif method == 'robust':
        return RobustScaler()
    else:
        raise ValueError(f"未支援的正規化方法: {method}")

def safe_divide(numerator, denominator, default_value=0):
    """安全除法，避免除零錯誤"""
    return np.where(denominator != 0, numerator / denominator, default_value)

def calculate_percentage_error(y_true, y_pred, epsilon=1e-8):
    """計算百分比誤差，避免除零"""
    return np.abs((y_true - y_pred) / (y_true + epsilon)) * 100

def format_number(number, decimal_places=4):
    """格式化數字顯示"""
    if isinstance(number, (int, float)):
        return f"{number:.{decimal_places}f}"
    return str(number)

def save_model(model: Any, filepath: str, model_name: str = "model") -> bool:
    """儲存訓練好的模型"""
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
        print(f"✅ 模型已儲存: {filepath}")
        return True
    except Exception as e:
        print(f"❌ 儲存模型失敗 {model_name}: {e}")
        return False

def load_model(filepath: str) -> Any:
    """載入訓練好的模型"""
    try:
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        print(f"✅ 模型已載入: {filepath}")
        return model
    except Exception as e:
        print(f"❌ 載入模型失敗: {e}")
        return None

def save_results_to_csv(results_list: List[Dict], filepath: str):
    """儲存結果到CSV檔案"""
    try:
        df = pd.DataFrame(results_list)
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        print(f"✅ 結果已儲存到: {filepath}")
        return True
    except Exception as e:
        print(f"❌ 儲存結果失敗: {e}")
        return False

def save_config_to_json(config_dict: Dict, filepath: str):
    """儲存設定到JSON檔案"""
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, ensure_ascii=False, indent=2)
        print(f"✅ 設定已儲存到: {filepath}")
        return True
    except Exception as e:
        print(f"❌ 儲存設定失敗: {e}")
        return False

def print_progress_bar(iteration: int, total: int, prefix: str = '', 
                      suffix: str = '', length: int = 50, fill: str = '█'):
    """顯示進度條"""
    percent = f"{100 * (iteration / float(total)):.1f}"
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='\r')
    if iteration == total:
        print()

def generate_timestamp():
    """產生時間戳記"""
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def create_experiment_directory(base_dir: str = None):
    """建立實驗專用資料夾"""
    if base_dir is None:
        base_dir = config.RESULTS_DIR
    
    timestamp = generate_timestamp()
    exp_dir = os.path.join(base_dir, f"experiment_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)
    
    # 建立子資料夾
    sub_dirs = ['models', 'plots', 'logs']
    for sub_dir in sub_dirs:
        os.makedirs(os.path.join(exp_dir, sub_dir), exist_ok=True)
    
    return exp_dir

def log_experiment_info(exp_dir: str, info_dict: Dict):
    """記錄實驗資訊"""
    log_file = os.path.join(exp_dir, "experiment_info.json")
    info_dict['timestamp'] = generate_timestamp()
    info_dict['config'] = {
        'train_ratio': config.TRAIN_RATIO,
        'val_ratio': config.VAL_RATIO,
        'test_ratio': config.TEST_RATIO,
        'random_state': config.RANDOM_STATE,
        'max_sample_size': config.MAX_SAMPLE_SIZE
    }
    save_config_to_json(info_dict, log_file)

def validate_data_shapes(X: np.ndarray, y: np.ndarray) -> bool:
    """驗證資料形狀是否正確"""
    if len(X) != len(y):
        print(f"❌ X和y的樣本數不匹配: {len(X)} vs {len(y)}")
        return False
    
    if len(X) == 0:
        print("❌ 資料集為空")
        return False
    
    print(f"✅ 資料驗證通過: {len(X)} 個樣本")
    return True

def check_missing_values(data: np.ndarray, name: str = "data") -> bool:
    """檢查缺失值"""
    if np.isnan(data).any():
        missing_count = np.isnan(data).sum()
        print(f"⚠️  {name} 包含 {missing_count} 個缺失值")
        return True
    return False

def handle_outliers(data: np.ndarray, method: str = 'iqr', 
                   factor: float = 1.5) -> np.ndarray:
    """處理異常值"""
    if method == 'iqr':
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        
        # 將異常值限制在邊界內
        return np.clip(data, lower_bound, upper_bound)
    
    elif method == 'zscore':
        mean = np.mean(data)
        std = np.std(data)
        z_scores = np.abs((data - mean) / std)
        return np.where(z_scores > factor, mean, data)
    
    else:
        return data

def split_time_series_data(X: np.ndarray, y: np.ndarray, 
                          train_ratio: float = 0.7, 
                          val_ratio: float = 0.15) -> Tuple[np.ndarray, ...]:
    """分割時間序列資料"""
    n_samples = len(X)
    
    # 計算分割點
    train_end = int(n_samples * train_ratio)
    val_end = int(n_samples * (train_ratio + val_ratio))
    
    # 分割資料
    X_train = X[:train_end]
    y_train = y[:train_end]
    X_val = X[train_end:val_end]
    y_val = y[train_end:val_end]
    X_test = X[val_end:]
    y_test = y[val_end:]
    
    print(f"📊 資料分割完成:")
    print(f"  訓練集: {len(X_train)} 樣本 ({len(X_train)/n_samples*100:.1f}%)")
    print(f"  驗證集: {len(X_val)} 樣本 ({len(X_val)/n_samples*100:.1f}%)")
    print(f"  測試集: {len(X_test)} 樣本 ({len(X_test)/n_samples*100:.1f}%)")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def print_section_header(title: str, width: int = 80, char: str = "="):
    """印出區段標題"""
    print(f"\n{char * width}")
    print(f"{title:^{width}}")
    print(f"{char * width}")

def print_subsection_header(title: str, width: int = 60, char: str = "-"):
    """印出子區段標題"""
    print(f"\n{char * width}")
    print(f"{title}")
    print(f"{char * width}")

class Timer:
    """計時器工具"""
    def __init__(self):
        self.start_time = None
        self.end_time = None
    
    def start(self):
        """開始計時"""
        self.start_time = datetime.datetime.now()
        return self.start_time
    
    def stop(self):
        """停止計時"""
        self.end_time = datetime.datetime.now()
        return self.end_time
    
    def elapsed(self) -> float:
        """取得經過時間（秒）"""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0
    
    def elapsed_str(self) -> str:
        """取得格式化的經過時間"""
        elapsed = self.elapsed()
        if elapsed < 60:
            return f"{elapsed:.2f} 秒"
        elif elapsed < 3600:
            minutes = elapsed // 60
            seconds = elapsed % 60
            return f"{int(minutes)} 分 {seconds:.1f} 秒"
        else:
            hours = elapsed // 3600
            minutes = (elapsed % 3600) // 60
            return f"{int(hours)} 時 {int(minutes)} 分"

def memory_usage():
    """取得記憶體使用情況"""
    try:
        import psutil
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        return f"{memory_mb:.1f} MB"
    except ImportError:
        return "N/A (需要 psutil)"

# 初始化 - 修正版
if __name__ != "__main__":  # 只有在被匯入時才執行
    set_random_seed(config.RANDOM_STATE)
    config.create_directories()