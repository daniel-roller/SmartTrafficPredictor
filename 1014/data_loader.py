# -*- coding: utf-8 -*-
"""
資料讀取與整合模組 (快速優化版)
"""

import pandas as pd
import numpy as np
import os
import glob
from typing import List, Dict, Tuple
from config import config

class DataLoader:
    """快速資料載入器"""
    
    def __init__(self):
        self.data_dir = config.DATA_DIR
        self.datasets = {}
        
    def load_all_datasets(self, sample_size=None) -> Dict[str, pd.DataFrame]:
        """載入所有 CSV 檔案 (限制資料量提升速度)"""
        if sample_size is None:
            sample_size = config.SAMPLE_SIZE
            
        csv_files = glob.glob(os.path.join(self.data_dir, "*.csv"))
        
        if not csv_files:
            raise FileNotFoundError(f"在 {self.data_dir} 中未找到 CSV 檔案")
        
        print(f"🚀 快速模式: 每個檔案最多載入 {sample_size} 筆資料")
        print(f"找到 {len(csv_files)} 個 CSV 檔案")
        
        for file_path in csv_files:
            dataset_name = os.path.splitext(os.path.basename(file_path))[0]
            try:
                # 使用 nrows 限制讀取行數
                df = pd.read_csv(file_path, encoding='utf-8', nrows=sample_size)
                df = self._standardize_columns(df)
                
                # 進一步隨機採樣
                if len(df) > sample_size:
                    df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
                
                self.datasets[dataset_name] = df
                print(f"✅ 載入成功: {dataset_name} ({len(df)} 筆資料)")
                
            except UnicodeDecodeError:
                try:
                    df = pd.read_csv(file_path, encoding='big5', nrows=sample_size)
                    df = self._standardize_columns(df)
                    
                    if len(df) > sample_size:
                        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
                    
                    self.datasets[dataset_name] = df
                    print(f"✅ 載入成功: {dataset_name} ({len(df)} 筆資料)")
                except Exception as e:
                    print(f"❌ 載入失敗 {dataset_name}: {e}")
            except Exception as e:
                print(f"❌ 載入失敗 {dataset_name}: {e}")
        
        return self.datasets
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """統一欄位名稱"""
        # 常見的時間欄位名稱
        time_columns = ['時間', 'datetime', 'timestamp', 'time', '日期時間', 'date', '日期']
        # 常見的流量欄位名稱  
        flow_columns = ['流量', 'traffic_flow', 'flow', 'volume', '車流量', '交通量']
        
        # 標準化時間欄位
        for col in df.columns:
            if any(time_col in col.lower() for time_col in [tc.lower() for tc in time_columns]):
                df = df.rename(columns={col: 'datetime'})
                break
        
        # 標準化流量欄位
        for col in df.columns:
            if any(flow_col in col.lower() for flow_col in [fc.lower() for fc in flow_columns]):
                df = df.rename(columns={col: 'traffic_flow'})
                break
        
        return df
    
    def get_dataset_info(self) -> pd.DataFrame:
        """獲取資料集資訊"""
        info_list = []
        for name, df in self.datasets.items():
            info_list.append({
                '資料集': name,
                '資料筆數': len(df),
                '欄位數': len(df.columns),
                '缺失值': df.isnull().sum().sum(),
                '時間範圍': f"{df['datetime'].min()} ~ {df['datetime'].max()}" if 'datetime' in df.columns else 'N/A'
            })
        return pd.DataFrame(info_list)
    
    def load_single_dataset_fast(self, dataset_name: str, sample_size: int = 5000) -> pd.DataFrame:
        """快速載入單一資料集"""
        csv_files = glob.glob(os.path.join(self.data_dir, f"*{dataset_name}*.csv"))
        
        if not csv_files:
            raise FileNotFoundError(f"找不到包含 {dataset_name} 的檔案")
        
        file_path = csv_files[0]
        df = pd.read_csv(file_path, encoding='utf-8', nrows=sample_size)
        df = self._standardize_columns(df)
        
        print(f"🚀 快速載入: {dataset_name} ({len(df)} 筆資料)")
        return df