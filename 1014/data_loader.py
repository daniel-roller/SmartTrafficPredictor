# -*- coding: utf-8 -*-
"""
交通流量預測系統 - CSV 數據載入器
從 select 資料夾讀取 CSV 檔案並轉換為模型需要的格式
"""

import numpy as np
import pandas as pd
import os
from typing import Dict, List, Tuple, Optional
from config import config

class CSVDataLoader:
    """CSV 數據載入器"""
    
    def __init__(self, csv_folder: str = None):
        """
        初始化 CSV 數據載入器
        
        Args:
            csv_folder: CSV 檔案所在資料夾，預設為 select/
        """
        if csv_folder is None:
            self.csv_folder = os.path.join(config.BASE_DIR, "select")
        else:
            self.csv_folder = csv_folder
            
        self.datasets = {}
        
    def load_csv_file(self, filepath: str) -> Optional[pd.DataFrame]:
        """
        載入單一 CSV 檔案
        
        Args:
            filepath: CSV 檔案路徑
            
        Returns:
            DataFrame 或 None (如果載入失敗)
        """
        try:
            df = pd.read_csv(filepath, encoding='utf-8')
            print(f"✅ 成功載入: {os.path.basename(filepath)}")
            print(f"   形狀: {df.shape}, 欄位: {list(df.columns)}")
            return df
        except Exception as e:
            print(f"❌ 載入失敗 {filepath}: {e}")
            return None
    
    def preprocess_traffic_data(self, df: pd.DataFrame, 
                                window_size: int = 12,
                                prediction_horizon: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        預處理交通數據，建立時間序列樣本
        
        Args:
            df: 原始 DataFrame
            window_size: 時間窗口大小 (用於預測的歷史資料點數)
            prediction_horizon: 預測時間間隔
            
        Returns:
            X (特徵), y (目標值)
        """
        # 確保 dataTime 欄位存在並轉換為時間格式
        if 'dataTime' in df.columns:
            df['dataTime'] = pd.to_datetime(df['dataTime'])
            df = df.sort_values('dataTime')
        
        # 選擇關鍵特徵
        feature_columns = []
        if 'Flow' in df.columns:
            feature_columns.append('Flow')
        if 'Speed' in df.columns:
            feature_columns.append('Speed')
        
        # 如果沒有這些欄位，嘗試使用所有數值欄位
        if not feature_columns:
            feature_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            # 排除 VD 和補值欄位
            feature_columns = [col for col in feature_columns 
                             if not col.startswith('VD') and col != '補值']
        
        if not feature_columns:
            raise ValueError("找不到可用的特徵欄位")
        
        print(f"   使用特徵: {feature_columns}")
        
        # 提取數據
        data = df[feature_columns].values
        
        # 建立時間窗口樣本
        X, y = [], []
        
        for i in range(len(data) - window_size - prediction_horizon + 1):
            # 輸入特徵：過去 window_size 個時間點的資料
            X.append(data[i:i + window_size])
            # 目標值：未來 prediction_horizon 時間點的 Flow
            if 'Flow' in feature_columns:
                flow_idx = feature_columns.index('Flow')
                y.append(data[i + window_size + prediction_horizon - 1, flow_idx])
            else:
                # 如果沒有 Flow，使用第一個特徵
                y.append(data[i + window_size + prediction_horizon - 1, 0])
        
        X = np.array(X)
        y = np.array(y).reshape(-1, 1)
        
        print(f"   生成樣本: X{X.shape}, y{y.shape}")
        
        return X, y
    
    def load_all_csv_datasets(self, window_size: int = 12, 
                             prediction_horizon: int = 1) -> Dict:
        """
        載入所有 CSV 檔案並轉換為訓練資料
        
        Args:
            window_size: 時間窗口大小
            prediction_horizon: 預測時間間隔
            
        Returns:
            字典，包含所有資料集的 X, y
        """
        if not os.path.exists(self.csv_folder):
            print(f"❌ 資料夾不存在: {self.csv_folder}")
            return {}
        
        csv_files = [f for f in os.listdir(self.csv_folder) if f.endswith('.csv')]
        
        if not csv_files:
            print(f"❌ 在 {self.csv_folder} 中找不到 CSV 檔案")
            return {}
        
        print(f"\n📂 找到 {len(csv_files)} 個 CSV 檔案")
        print(f"📊 時間窗口: {window_size}, 預測間隔: {prediction_horizon}")
        print("-" * 60)
        
        datasets = {}
        
        for csv_file in csv_files:
            filepath = os.path.join(self.csv_folder, csv_file)
            dataset_name = os.path.splitext(csv_file)[0]
            
            print(f"\n🔄 處理: {dataset_name}")
            
            # 載入 CSV
            df = self.load_csv_file(filepath)
            if df is None:
                continue
            
            try:
                # 預處理並建立時間序列樣本
                X, y = self.preprocess_traffic_data(df, window_size, prediction_horizon)
                
                # 限制樣本數量
                if len(X) > config.MAX_SAMPLE_SIZE:
                    print(f"   ⚠️  樣本數過多，採樣前 {config.MAX_SAMPLE_SIZE} 個")
                    X = X[:config.MAX_SAMPLE_SIZE]
                    y = y[:config.MAX_SAMPLE_SIZE]
                
                datasets[dataset_name] = {
                    'X': X,
                    'y': y,
                    'name': dataset_name,
                    'original_shape': X.shape,
                    'samples': len(X),
                    'source_file': csv_file
                }
                
                print(f"   ✅ 成功處理: {len(X)} 個樣本")
                
            except Exception as e:
                print(f"   ❌ 處理失敗: {e}")
                continue
        
        print(f"\n{'='*60}")
        print(f"✅ 總共載入 {len(datasets)} 個資料集")
        
        return datasets
    
    def save_datasets_as_npy(self, datasets: Dict, output_dir: str = None):
        """
        將資料集儲存為 .npy 格式 (可選)
        
        Args:
            datasets: 資料集字典
            output_dir: 輸出資料夾
        """
        if output_dir is None:
            output_dir = os.path.join(config.BASE_DIR, "datasets")
        
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n💾 儲存資料集到: {output_dir}")
        
        for name, data in datasets.items():
            try:
                X_path = os.path.join(output_dir, f"{name}_X.npy")
                y_path = os.path.join(output_dir, f"{name}_y.npy")
                
                np.save(X_path, data['X'])
                np.save(y_path, data['y'])
                
                print(f"   ✅ {name}: X 和 y 已儲存")
                
            except Exception as e:
                print(f"   ❌ 儲存 {name} 失敗: {e}")
        
        print("✅ 所有資料集已儲存")
    
    def get_dataset_summary(self, datasets: Dict) -> pd.DataFrame:
        """
        取得資料集摘要資訊
        
        Args:
            datasets: 資料集字典
            
        Returns:
            摘要 DataFrame
        """
        summary_data = []
        
        for name, data in datasets.items():
            summary_data.append({
                '資料集名稱': name,
                '樣本數': data['samples'],
                'X 形狀': str(data['X'].shape),
                'y 形狀': str(data['y'].shape),
                '來源檔案': data['source_file']
            })
        
        return pd.DataFrame(summary_data)


def main():
    """測試 CSV 載入器"""
    print("🚗 交通流量 CSV 數據載入器")
    print("=" * 60)
    
    # 建立載入器
    loader = CSVDataLoader()
    
    # 載入所有 CSV 檔案
    datasets = loader.load_all_csv_datasets(window_size=12, prediction_horizon=1)
    
    if datasets:
        # 顯示摘要
        print("\n📊 資料集摘要:")
        summary = loader.get_dataset_summary(datasets)
        print(summary.to_string(index=False))
        
        # 可選：儲存為 .npy 格式
        save_choice = input("\n是否儲存為 .npy 格式? (y/n): ").strip().lower()
        if save_choice == 'y':
            loader.save_datasets_as_npy(datasets)
    else:
        print("\n❌ 沒有成功載入任何資料集")


if __name__ == "__main__":
    main()
