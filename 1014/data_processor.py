# -*- coding: utf-8 -*-
"""
交通流量預測系統 - 資料處理器 (簡化版 - 支援 CSV)
"""

import numpy as np
import pandas as pd
import os
import datetime
from typing import Dict, List, Tuple, Optional, Union
from sklearn.preprocessing import StandardScaler

from config import config
from data_loader import CSVDataLoader
from utils import (
    validate_data_shapes, check_missing_values, handle_outliers,
    split_time_series_data, create_scaler, print_subsection_header,
    Timer
)

class SimpleFeatureEngineering:
    """簡化特徵工程"""
    
    def __init__(self):
        self.timer = Timer()
    
    def extract_basic_features(self, X: np.ndarray) -> np.ndarray:
        """提取基本統計特徵"""
        print("📊 提取基本特徵...")
        self.timer.start()
        
        basic_features = []
        for i in range(X.shape[0]):
            sample = X[i].ravel()
            
            # 只保留最基本的統計量
            features = [
                np.mean(sample),      # 平均值
                np.std(sample),       # 標準差
                np.min(sample),       # 最小值
                np.max(sample),       # 最大值
                np.median(sample),    # 中位數
                np.sum(sample)        # 總和
            ]
            basic_features.append(features)
        
        self.timer.stop()
        print(f"   ✅ 基本特徵提取完成 ({self.timer.elapsed_str()})")
        
        return np.array(basic_features)
    
    def create_simple_lag_features(self, y: np.ndarray) -> np.ndarray:
        """建立簡單滯後特徵"""
        lags = config.LAG_FEATURES
        print(f"🔄 建立滯後特徵: {lags}")
        self.timer.start()
        
        lag_features = []
        y_flat = y.ravel()
        
        for i in range(len(y_flat)):
            features = []
            for lag in lags:
                if i >= lag:
                    features.append(y_flat[i - lag])
                else:
                    features.append(0.0)
            lag_features.append(features)
        
        self.timer.stop()
        print(f"   ✅ 滯後特徵建立完成 ({self.timer.elapsed_str()})")
        
        return np.array(lag_features)
    
    def engineer_features(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """簡化特徵工程"""
        print_subsection_header("🔧 開始簡化特徵工程")
        
        # 原始特徵（攤平）
        original_features = X.reshape(X.shape[0], -1)
        print(f"✅ 原始特徵: {original_features.shape[1]} 維")
        
        # 基本統計特徵
        basic_features = self.extract_basic_features(X)
        print(f"✅ 基本特徵: {basic_features.shape[1]} 維")
        
        # 滯後特徵
        lag_features = self.create_simple_lag_features(y)
        print(f"✅ 滯後特徵: {lag_features.shape[1]} 維")
        
        # 組合特徵
        enhanced_features = np.column_stack([
            original_features,
            basic_features,
            lag_features
        ])
        
        total_features = enhanced_features.shape[1]
        print(f"\n🎯 總特徵數: {total_features} 維")
        
        return enhanced_features

class DataProcessor:
    """資料處理主類 - 支援 CSV 直接載入"""
    
    def __init__(self):
        self.feature_engineer = SimpleFeatureEngineering()
        self.scaler = create_scaler(config.SCALER_METHOD)
        self.datasets: Dict = {}
        self.csv_loader = CSVDataLoader()
    
    def load_all_datasets(self) -> Dict:
        """載入所有資料集 - 從 CSV 或 .npy 檔案"""
        print_subsection_header("📂 載入交通資料集")
        
        # 優先從 CSV 載入
        csv_folder = os.path.join(config.BASE_DIR, "select")
        if os.path.exists(csv_folder):
            csv_files = [f for f in os.listdir(csv_folder) if f.endswith('.csv')]
            if csv_files:
                print("📊 從 CSV 檔案載入資料...")
                datasets = self.csv_loader.load_all_csv_datasets(window_size=12, prediction_horizon=1)
                
                if datasets:
                    self.datasets = datasets
                    print(f"\n✅ 從 CSV 成功載入 {len(datasets)} 個資料集")
                    return datasets
        
        # 如果沒有 CSV，嘗試從 .npy 載入
        if hasattr(config, 'DATASETS_INFO') and config.DATASETS_INFO:
            print("📊 從 .npy 檔案載入資料...")
            datasets = {}
            for dataset_name in config.DATASETS_INFO.keys():
                dataset = self.load_single_dataset_from_npy(dataset_name)
                if dataset is not None:
                    datasets[dataset_name] = dataset
            
            if datasets:
                self.datasets = datasets
                print(f"\n✅ 從 .npy 成功載入 {len(datasets)} 個資料集")
                return datasets
        
        print("❌ 沒有成功載入任何資料集")
        return {}
    
    def load_single_dataset_from_npy(self, dataset_name: str) -> Optional[Dict]:
        """從 .npy 檔案載入單一資料集"""
        if dataset_name not in config.DATASETS_INFO:
            print(f"❌ 未知的資料集: {dataset_name}")
            return None
        
        files = config.DATASETS_INFO[dataset_name]
        X_path = os.path.join(config.DATASETS_DIR, files['X'])
        y_path = os.path.join(config.DATASETS_DIR, files['y'])
        
        try:
            if not os.path.exists(X_path) or not os.path.exists(y_path):
                print(f"❌ 找不到檔案: {dataset_name}")
                return None
            
            X = np.load(X_path)
            y = np.load(y_path)
            
            # 限制樣本數量
            if len(X) > config.MAX_SAMPLE_SIZE:
                print(f"⚠️  資料集 {dataset_name} 過大，採樣前 {config.MAX_SAMPLE_SIZE} 個樣本")
                X = X[:config.MAX_SAMPLE_SIZE]
                y = y[:config.MAX_SAMPLE_SIZE]
            
            # 驗證資料
            if not validate_data_shapes(X, y):
                return None
            
            # 檢查缺失值
            check_missing_values(X, f"{dataset_name}_X")
            check_missing_values(y, f"{dataset_name}_y")
            
            dataset = {
                'X': X,
                'y': y,
                'name': dataset_name,
                'original_shape': X.shape,
                'samples': len(X)
            }
            
            print(f"✅ {dataset_name}: X{X.shape}, y{y.shape}")
            return dataset
            
        except Exception as e:
            print(f"❌ 載入 {dataset_name} 失敗: {e}")
            return None
    
    def process_single_dataset(self, dataset: Dict) -> Dict:
        """處理單一資料集"""
        dataset_name = dataset['name']
        X, y = dataset['X'], dataset['y']
        
        print(f"\n🔄 處理資料集: {dataset_name}")
        
        # 簡化特徵工程
        enhanced_X = self.feature_engineer.engineer_features(X, y)
        
        # 分割資料
        X_train, X_val, X_test, y_train, y_val, y_test = split_time_series_data(
            enhanced_X, y, 
            train_ratio=config.TRAIN_RATIO,
            val_ratio=config.VAL_RATIO
        )
        
        # 正規化特徵
        print("🔧 正規化特徵...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        processed_data = {
            'name': dataset_name,
            'X_train': X_train_scaled,
            'X_val': X_val_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'scaler': self.scaler,
            'original_shape': dataset['original_shape'],
            'enhanced_shape': enhanced_X.shape,
            'feature_count': enhanced_X.shape[1]
        }
        
        print(f"✅ {dataset_name} 處理完成")
        
        return processed_data
    
    def process_all_datasets(self) -> Dict:
        """處理所有資料集"""
        print_subsection_header("⚙️ 資料預處理")
        
        if not self.datasets:
            print("❌ 沒有可用的資料集，請先載入資料")
            return {}
        
        processed_datasets = {}
        
        for dataset_name, dataset in self.datasets.items():
            try:
                processed_data = self.process_single_dataset(dataset)
                processed_datasets[dataset_name] = processed_data
            except Exception as e:
                print(f"❌ 處理 {dataset_name} 失敗: {e}")
        
        print(f"\n✅ 資料預處理完成，共處理 {len(processed_datasets)} 個資料集")
        return processed_datasets