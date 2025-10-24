# -*- coding: utf-8 -*-
"""
特徵工程模組 (增強版)
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import List
from config import config
from utils import is_holiday, is_peak_hour

class FeatureEngineer:
    """增強版特徵工程器"""
    
    def __init__(self):
        pass
    
    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """創建時間特徵 (增強版)"""
        df_feat = df.copy()
        
        if 'datetime' not in df_feat.columns:
            raise ValueError("需要 datetime 欄位")
        
        print("   📅 處理時間特徵...")
        
        # 基本時間特徵
        df_feat['hour'] = df_feat['datetime'].dt.hour
        df_feat['day_of_week'] = df_feat['datetime'].dt.dayofweek
        df_feat['month'] = df_feat['datetime'].dt.month
        df_feat['day_of_month'] = df_feat['datetime'].dt.day
        df_feat['week_of_year'] = df_feat['datetime'].dt.isocalendar().week
        
        # 週期性特徵（正弦余弦編碼）
        df_feat['hour_sin'] = np.sin(2 * np.pi * df_feat['hour'] / 24)
        df_feat['hour_cos'] = np.cos(2 * np.pi * df_feat['hour'] / 24)
        df_feat['dow_sin'] = np.sin(2 * np.pi * df_feat['day_of_week'] / 7)
        df_feat['dow_cos'] = np.cos(2 * np.pi * df_feat['day_of_week'] / 7)
        df_feat['month_sin'] = np.sin(2 * np.pi * (df_feat['month'] - 1) / 12)
        df_feat['month_cos'] = np.cos(2 * np.pi * (df_feat['month'] - 1) / 12)
        
        print("   ✅ 時間特徵完成")
        return df_feat
    
    def create_traffic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """創建交通相關特徵 (增強版)"""
        df_feat = df.copy()
        
        print("   🚦 處理交通特徵...")
        
        # 假日特徵（簡化版以提升速度）
        if 'day_of_week' in df_feat.columns:
            df_feat['is_holiday'] = (df_feat['day_of_week'] >= 5).astype(int)
        
        # 尖峰時段特徵
        if 'hour' in df_feat.columns:
            df_feat['is_peak_hour'] = df_feat['hour'].apply(is_peak_hour).astype(int)
            
            # 更細緻的時段分類
            df_feat['time_period'] = df_feat['hour'].apply(self._categorize_time_period)
            
            # 工作時間
            df_feat['is_work_hour'] = (
                (df_feat['hour'] >= 6) & (df_feat['hour'] <= 22)
            ).astype(int)
            
            # 深夜時段
            df_feat['is_night'] = (
                (df_feat['hour'] >= 22) | (df_feat['hour'] <= 5)
            ).astype(int)
        
        # 工作日/週末
        if 'day_of_week' in df_feat.columns:
            df_feat['is_weekend'] = (df_feat['day_of_week'] >= 5).astype(int)
            df_feat['is_friday'] = (df_feat['day_of_week'] == 4).astype(int)
            df_feat['is_monday'] = (df_feat['day_of_week'] == 0).astype(int)
        
        print("   ✅ 交通特徵完成")
        return df_feat
    
    def _categorize_time_period(self, hour):
        """時段分類"""
        if 5 <= hour < 9:
            return 1  # 早高峰
        elif 9 <= hour < 17:
            return 2  # 日間
        elif 17 <= hour < 20:
            return 3  # 晚高峰
        elif 20 <= hour < 23:
            return 4  # 晚間
        else:
            return 0  # 深夜
    
    def create_lag_features(self, df: pd.DataFrame, target_col: str = 'traffic_flow') -> pd.DataFrame:
        """創建滯後特徵 (增強版)"""
        df_feat = df.copy()
        
        if target_col not in df_feat.columns:
            return df_feat
        
        print("   ⏳ 處理滯後特徵...")
        
        for lag in config.LAG_FEATURES:
            df_feat[f'{target_col}_lag_{lag}'] = df_feat[target_col].shift(lag)
            print(f"     創建 lag_{lag}")
        
        print("   ✅ 滯後特徵完成")
        return df_feat
    
    def create_rolling_features(self, df: pd.DataFrame, target_col: str = 'traffic_flow') -> pd.DataFrame:
        """創建滾動統計特徵 (增強版)"""
        df_feat = df.copy()
        
        if target_col not in df_feat.columns:
            return df_feat
        
        print("   📊 處理滾動特徵...")
        
        for window in config.ROLLING_WINDOWS:
            df_feat[f'{target_col}_rolling_mean_{window}'] = df_feat[target_col].rolling(window).mean()
            df_feat[f'{target_col}_rolling_std_{window}'] = df_feat[target_col].rolling(window).std()
            df_feat[f'{target_col}_rolling_max_{window}'] = df_feat[target_col].rolling(window).max()
            df_feat[f'{target_col}_rolling_min_{window}'] = df_feat[target_col].rolling(window).min()
            print(f"     創建 rolling_{window}")
        
        print("   ✅ 滾動特徵完成")
        return df_feat
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """創建交互特徵"""
        df_feat = df.copy()
        
        print("   🔄 處理交互特徵...")
        
        # 時間與週末的交互作用
        if 'hour' in df_feat.columns and 'is_weekend' in df_feat.columns:
            df_feat['hour_weekend'] = df_feat['hour'] * df_feat['is_weekend']
        
        # 尖峰時段與工作日的交互作用
        if 'is_peak_hour' in df_feat.columns and 'is_weekend' in df_feat.columns:
            df_feat['peak_weekday'] = df_feat['is_peak_hour'] * (1 - df_feat['is_weekend'])
        
        print("   ✅ 交互特徵完成")
        return df_feat
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """完整特徵工程流程 (增強版)"""
        print("🚀 開始增強版特徵工程...")
        
        # 基本時間特徵
        df_engineered = self.create_time_features(df)
        
        # 交通相關特徵
        df_engineered = self.create_traffic_features(df_engineered)
        
        # 滯後特徵
        if config.LAG_FEATURES:
            df_engineered = self.create_lag_features(df_engineered)
        
        # 滾動統計特徵
        if config.ROLLING_WINDOWS:
            df_engineered = self.create_rolling_features(df_engineered)
        
        # 交互特徵
        df_engineered = self.create_interaction_features(df_engineered)
        
        # 移除缺失值
        original_len = len(df_engineered)
        df_engineered = df_engineered.dropna()
        dropped = original_len - len(df_engineered)
        
        print(f"✅ 增強版特徵工程完成:")
        print(f"   📊 特徵數量: {len(df_engineered.columns)}")
        print(f"   📈 資料筆數: {len(df_engineered)} (丟棄 {dropped} 筆缺失值)")
        
        return df_engineered