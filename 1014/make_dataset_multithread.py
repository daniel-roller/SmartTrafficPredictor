import os
import pandas as pd
import numpy as np
import pickle
from config import Config
from segment_selector import analyze_cleaned_data

class SimpleDatasetMaker:
    """簡化版資料集製作器"""
    
    def __init__(self):
        self.config = Config()
        print("🔧 資料集製作器初始化完成")
    
    def prepare_data(self, df, time_col, speed_col):
        """準備訓練資料"""
        # 確保時間順序
        df = df.sort_values(time_col).reset_index(drop=True)
        
        # 基本特徵
        df['hour'] = df[time_col].dt.hour
        df['weekday'] = df[time_col].dt.weekday
        df['is_weekend'] = (df['weekday'] >= 5).astype(int)
        df['is_peak'] = ((df['hour'].between(7, 9)) | (df['hour'].between(17, 19))).astype(int)
        
        # 檢查是否有flow欄位
        flow_col = None
        for col in df.columns:
            if 'flow' in col.lower() or 'vehicle' in col.lower():
                flow_col = col
                break
        
        if flow_col:
            features = [speed_col, flow_col, 'hour', 'weekday', 'is_weekend', 'is_peak']
            targets = [speed_col, flow_col]
        else:
            # 只有速度資料
            features = [speed_col, 'hour', 'weekday', 'is_weekend', 'is_peak'] 
            targets = [speed_col]
        
        return df[features].values, df[targets].values, features, targets
    
    def create_sequences(self, X, y, window_size, horizon):
        """建立時間序列"""
        sequences_X, sequences_y = [], []
        
        for i in range(window_size, len(X) - horizon + 1):
            sequences_X.append(X[i-window_size:i])
            sequences_y.append(y[i+horizon-1])
        
        return np.array(sequences_X), np.array(sequences_y)
    
    def process_segment(self, segment_info):
        """處理單一路段"""
        segment_name = segment_info['segment']
        file_path = segment_info['file_path']
        time_col = segment_info['time_col']
        speed_col = segment_info['speed_col']
        
        print(f"📊 處理路段: {segment_name}")
        
        try:
            # 載入資料
            if file_path.endswith('.parquet'):
                df = pd.read_parquet(file_path)
            else:
                df = pd.read_csv(file_path)
            
            # 標準化欄位名稱
            df.columns = [col.strip().lower() for col in df.columns]
            
            # 轉換時間格式
            df[time_col] = pd.to_datetime(df[time_col])
            
            # 移除缺值
            df = df.dropna(subset=[speed_col])
            
            # 準備特徵和目標
            X_data, y_data, feature_names, target_names = self.prepare_data(df, time_col, speed_col)
            
            # 建立序列
            X_seq, y_seq = self.create_sequences(X_data, y_data, 
                                               self.config.WINDOW_SIZE, 
                                               self.config.HORIZON)
            
            if len(X_seq) == 0:
                print(f"⚠️ {segment_name} 資料不足，跳過")
                return None
            
            # 分割資料
            train_size = int(len(X_seq) * self.config.TRAIN_RATIO)
            val_size = int(len(X_seq) * self.config.VAL_RATIO)
            
            X_train = X_seq[:train_size]
            X_val = X_seq[train_size:train_size+val_size]
            X_test = X_seq[train_size+val_size:]
            
            y_train = y_seq[:train_size]
            y_val = y_seq[train_size:train_size+val_size]
            y_test = y_seq[train_size+val_size:]
            
            # 標準化
            X_mean = X_train.mean(axis=(0, 1), keepdims=True)
            X_std = X_train.std(axis=(0, 1), keepdims=True) + 1e-8
            y_mean = y_train.mean(axis=0, keepdims=True)
            y_std = y_train.std(axis=0, keepdims=True) + 1e-8
            
            X_train = (X_train - X_mean) / X_std
            X_val = (X_val - X_mean) / X_std
            X_test = (X_test - X_mean) / X_std
            
            y_train = (y_train - y_mean) / y_std
            y_val = (y_val - y_mean) / y_std
            y_test = (y_test - y_mean) / y_std
            
            # 建立資料集
            dataset = {
                'X_train': X_train, 'y_train': y_train,
                'X_val': X_val, 'y_val': y_val,
                'X_test': X_test, 'y_test': y_test,
                'X_mean': X_mean, 'X_std': X_std,
                'y_mean': y_mean, 'y_std': y_std,
                'segment_name': segment_name,
                'window_size': self.config.WINDOW_SIZE,
                'horizon': self.config.HORIZON,
                'features': feature_names,
                'targets': target_names
            }
            
            # 儲存
            save_path = os.path.join(self.config.DATASETS_DIR, f"{segment_name}.pkl")
            with open(save_path, 'wb') as f:
                pickle.dump(dataset, f)
            
            print(f"✅ {segment_name} 完成 - 訓練: {len(X_train)}, 驗證: {len(X_val)}, 測試: {len(X_test)}")
            return save_path
            
        except Exception as e:
            print(f"❌ {segment_name} 失敗: {e}")
            return None
    
    def create_all_datasets(self):
        """建立所有資料集"""
        # 分析並選擇路段
        selected_names, segments_info = analyze_cleaned_data()
        
        if not selected_names:
            print("❌ 沒有可用的路段")
            return
        
        print(f"\n🚀 開始處理 {len(selected_names)} 個路段...")
        
        successful = []
        failed = []
        
        for _, segment_info in segments_info.iterrows():
            result = self.process_segment(segment_info)
            if result:
                successful.append(segment_info['segment'])
            else:
                failed.append(segment_info['segment'])
        
        print(f"\n📊 資料集建立完成:")
        print(f"  ✅ 成功: {len(successful)} 個路段")
        print(f"  ❌ 失敗: {len(failed)} 個路段")
        
        if successful:
            print(f"  📦 成功路段: {successful}")
        if failed:
            print(f"  ⚠️ 失敗路段: {failed}")

def main():
    maker = SimpleDatasetMaker()
    maker.create_all_datasets()

if __name__ == "__main__":
    main()