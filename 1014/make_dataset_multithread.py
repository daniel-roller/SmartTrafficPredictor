import os
import pandas as pd
import numpy as np
import pickle
from config import Config
from enhanced_feature_engineering import EnhancedFeatureEngineer
from segment_selector import analyze_cleaned_data

class EnhancedDatasetMaker:
    """增強版資料集製作器"""
    
    def __init__(self):
        self.config = Config()
        self.feature_engineer = EnhancedFeatureEngineer()
        print("🔧 增強版資料集製作器初始化完成")
    
    def prepare_enhanced_data(self, df, time_col, speed_col, vehicle_col=None):
        """準備增強特徵資料"""
        print("🚀 開始特徵工程...")
        
        # 應用增強特徵工程
        df_enhanced = self.feature_engineer.create_all_features(
            df, time_col, speed_col, vehicle_col
        )
        
        # 選擇特徵 (排除不需要的欄位)
        exclude_cols = [time_col, 'vd_id', 'year', 'day', 'season']  # 排除不適合的特徵
        feature_cols = [col for col in df_enhanced.columns if col not in exclude_cols]
        
        # 確定目標變數
        if vehicle_col and vehicle_col in df_enhanced.columns:
            target_cols = [speed_col, vehicle_col]
        else:
            target_cols = [speed_col]
        
        # 特徵選擇 (移除目標變數)
        feature_cols = [col for col in feature_cols if col not in target_cols]
        
        print(f"📊 選擇 {len(feature_cols)} 個特徵, {len(target_cols)} 個目標")
        print(f"🎯 目標變數: {target_cols}")
        
        return df_enhanced[feature_cols].values, df_enhanced[target_cols].values, feature_cols, target_cols
    
    def create_sequences(self, X, y, window_size, horizon):
        """建立時間序列"""
        sequences_X, sequences_y = [], []
        
        for i in range(window_size, len(X) - horizon + 1):
            sequences_X.append(X[i-window_size:i])
            sequences_y.append(y[i+horizon-1])
        
        return np.array(sequences_X), np.array(sequences_y)
    
    def process_segment(self, segment_info):
        """處理單一路段 - 增強版"""
        segment_name = segment_info['segment']
        file_path = segment_info['file_path']
        time_col = segment_info['time_col']
        speed_col = segment_info['speed_col']
        
        print(f"\n📊 處理路段: {segment_name}")
        
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
            
            # 檢查車流量欄位
            vehicle_col = None
            for col in df.columns:
                if 'flow' in col.lower() or 'vehicle' in col.lower():
                    vehicle_col = col
                    break
            
            # 應用增強特徵工程
            X_data, y_data, feature_names, target_names = self.prepare_enhanced_data(
                df, time_col, speed_col, vehicle_col
            )
            
            # 建立序列
            X_seq, y_seq = self.create_sequences(X_data, y_data, 
                                               self.config.WINDOW_SIZE, 
                                               self.config.HORIZON)
            
            if len(X_seq) == 0:
                print(f"⚠️ {segment_name} 資料不足，跳過")
                return None
            
            print(f"📈 序列資料: {X_seq.shape}, 目標: {y_seq.shape}")
            
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
                'targets': target_names,
                'enhanced_version': True  # 標記為增強版
            }
            
            # 儲存
            save_path = os.path.join(self.config.DATASETS_DIR, f"{segment_name}_enhanced.pkl")
            with open(save_path, 'wb') as f:
                pickle.dump(dataset, f)
            
            print(f"✅ {segment_name} 完成")
            print(f"   📊 訓練: {len(X_train)}, 驗證: {len(X_val)}, 測試: {len(X_test)}")
            print(f"   🚀 特徵數量: {len(feature_names)}")
            print(f"   💾 儲存至: {save_path}")
            
            return save_path
            
        except Exception as e:
            print(f"❌ {segment_name} 失敗: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def create_all_datasets(self):
        """建立所有增強版資料集"""
        # 分析並選擇路段
        selected_names, segments_info = analyze_cleaned_data()
        
        if not selected_names:
            print("❌ 沒有可用的路段")
            return
        
        print(f"\n🚀 開始處理 {len(selected_names)} 個路段 (增強特徵版)...")
        
        successful = []
        failed = []
        
        for _, segment_info in segments_info.iterrows():
            result = self.process_segment(segment_info)
            if result:
                successful.append(segment_info['segment'])
            else:
                failed.append(segment_info['segment'])
        
        print(f"\n📊 增強版資料集建立完成:")
        print(f"  ✅ 成功: {len(successful)} 個路段")
        print(f"  ❌ 失敗: {len(failed)} 個路段")
        
        if successful:
            print(f"  📦 成功路段: {successful}")
        if failed:
            print(f"  ⚠️ 失敗路段: {failed}")

def main():
    maker = EnhancedDatasetMaker()
    maker.create_all_datasets()

if __name__ == "__main__":
    main()