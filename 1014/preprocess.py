# -*- coding: utf-8 -*-
"""
資料前處理模組 (SVM優化版)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer
from typing import Tuple, Optional
from config import config

class DataPreprocessor:
    """SVM優化版資料前處理器"""
    
    def __init__(self):
        self.scaler = None
        self.svm_scaler = None
        self.imputer = None
        
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """清洗資料 (加強版)"""
        print("🧹 開始資料清洗...")
        
        df_clean = df.copy()
        
        # 處理時間欄位
        if 'datetime' in df_clean.columns:
            df_clean['datetime'] = pd.to_datetime(df_clean['datetime'], errors='coerce')
            df_clean = df_clean.dropna(subset=['datetime'])
            df_clean = df_clean.sort_values('datetime').reset_index(drop=True)
        
        # 處理交通流量欄位
        if 'traffic_flow' in df_clean.columns:
            # 移除明顯異常值（使用四分位數方法）
            Q1 = df_clean['traffic_flow'].quantile(0.25)
            Q3 = df_clean['traffic_flow'].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # 記錄異常值數量
            outliers = df_clean[(df_clean['traffic_flow'] < lower_bound) | 
                              (df_clean['traffic_flow'] > upper_bound)]
            print(f"   🚫 移除 {len(outliers)} 個異常值")
            
            # 移除異常值
            df_clean = df_clean[(df_clean['traffic_flow'] >= lower_bound) & 
                              (df_clean['traffic_flow'] <= upper_bound)]
            
            # 移除負值和零值
            original_len = len(df_clean)
            df_clean = df_clean[df_clean['traffic_flow'] > 0]
            removed = original_len - len(df_clean)
            if removed > 0:
                print(f"   🚫 移除 {removed} 個負值或零值")
        
        # 處理缺失值
        missing_before = df_clean.isnull().sum().sum()
        if missing_before > 0:
            print(f"   🔧 處理 {missing_before} 個缺失值")
            df_clean = df_clean.dropna()
        
        print(f"✅ 資料清洗完成: {len(df_clean)} 筆資料")
        return df_clean
    
    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """分割資料集 (時間序列版)"""
        print("📊 分割資料集...")
        
        # 按時間順序分割
        n = len(df)
        train_end = int(n * config.TRAIN_SIZE / (config.TRAIN_SIZE + config.VAL_SIZE + config.TEST_SIZE) * (config.TRAIN_SIZE + config.VAL_SIZE + config.TEST_SIZE))
        val_end = train_end + int(n * config.VAL_SIZE / (config.TRAIN_SIZE + config.VAL_SIZE + config.TEST_SIZE) * (config.TRAIN_SIZE + config.VAL_SIZE + config.TEST_SIZE))
        
        train_data = df.iloc[:train_end].copy()
        val_data = df.iloc[train_end:val_end].copy()
        test_data = df.iloc[val_end:].copy()
        
        print(f"   📈 訓練集: {len(train_data)} 筆")
        print(f"   📊 驗證集: {len(val_data)} 筆")
        print(f"   🎯 測試集: {len(test_data)} 筆")
        
        return train_data, val_data, test_data
    
    def scale_features(self, X_train, X_test=None, method='standard'):
        """標準特徵縮放 (一般模型用)"""
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
        elif method == 'robust':
            self.scaler = RobustScaler()
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
            return X_train_scaled, X_test_scaled
        
        return X_train_scaled
    
    def scale_features_for_svm(self, X_train, X_test=None):
        """SVM專用特徵縮放 (使用RobustScaler對抗異常值)"""
        print("🔧 SVM專用特徵縮放...")
        
        # 使用RobustScaler，對異常值更穩定
        self.svm_scaler = RobustScaler()
        
        X_train_scaled = self.svm_scaler.fit_transform(X_train)
        
        # 進一步確保數據在合理範圍內
        X_train_scaled = np.clip(X_train_scaled, -5, 5)
        
        if X_test is not None:
            X_test_scaled = self.svm_scaler.transform(X_test)
            X_test_scaled = np.clip(X_test_scaled, -5, 5)
            return X_train_scaled, X_test_scaled
        
        return X_train_scaled
    
    def prepare_data_for_model(self, df: pd.DataFrame, model_type: str):
        """為特定模型準備資料"""
        feature_columns = [col for col in df.columns 
                         if col not in ['datetime', 'traffic_flow', 'dataset_source']]
        
        train_data, val_data, test_data = self.split_data(df)
        
        X_train = train_data[feature_columns].values
        y_train = train_data['traffic_flow'].values
        X_test = test_data[feature_columns].values
        y_test = test_data['traffic_flow'].values
        
        if model_type == 'SVM':
            # SVM使用專用縮放
            X_train_scaled, X_test_scaled = self.scale_features_for_svm(X_train, X_test)
        else:
            # 其他模型使用標準縮放
            X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test, method='standard')
        
        return X_train_scaled, y_train, X_test_scaled, y_test, feature_columns