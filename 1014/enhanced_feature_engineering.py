import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime
try:
    import holidays
except ImportError:
    print("⚠️ 未安裝 holidays 套件，將跳過節假日特徵")
    holidays = None

class EnhancedFeatureEngineer:
    """增強版特徵工程器"""
    
    def __init__(self):
        # 台灣節假日
        if holidays:
            try:
                self.taiwan_holidays = holidays.Taiwan(years=range(2020, 2025))
            except:
                self.taiwan_holidays = {}
        else:
            self.taiwan_holidays = {}
        
        # 尖峰時段定義
        self.morning_rush = (7, 9)    # 早上尖峰 7-9點
        self.evening_rush = (17, 19)  # 晚上尖峰 17-19點
        self.noon_rush = (11, 13)     # 午餐時間 11-13點
        
        # 季節定義 (台灣氣候)
        self.seasons = {
            'spring': [3, 4, 5],      # 春季
            'summer': [6, 7, 8],      # 夏季  
            'autumn': [9, 10, 11],    # 秋季
            'winter': [12, 1, 2]      # 冬季
        }
    
    def add_time_features(self, df, time_col='time_bin'):
        """添加基礎時間特徵"""
        df = df.copy()
        
        # 基本時間特徵
        df['year'] = df[time_col].dt.year
        df['month'] = df[time_col].dt.month
        df['day'] = df[time_col].dt.day
        df['hour'] = df[time_col].dt.hour
        df['weekday'] = df[time_col].dt.weekday  # 0=Monday, 6=Sunday
        df['day_of_year'] = df[time_col].dt.dayofyear
        
        return df
    
    def add_calendar_features(self, df, time_col='time_bin'):
        """添加日曆相關特徵"""
        df = df.copy()
        
        # 週末/平日
        df['is_weekend'] = (df['weekday'] >= 5).astype(int)
        df['is_weekday'] = (df['weekday'] < 5).astype(int)
        
        # 節假日 
        try:
            if self.taiwan_holidays:
                df['is_holiday'] = df[time_col].dt.date.apply(lambda x: x in self.taiwan_holidays).astype(int)
                # 節假日前一天
                df['is_pre_holiday'] = df[time_col].dt.date.apply(
                    lambda x: (x + pd.Timedelta(days=1)) in self.taiwan_holidays
                ).astype(int)
            else:
                df['is_holiday'] = 0
                df['is_pre_holiday'] = 0
        except:
            df['is_holiday'] = 0
            df['is_pre_holiday'] = 0
        
        # 月初/月底 (薪資發放影響)
        df['is_month_start'] = (df['day'] <= 5).astype(int)
        df['is_month_end'] = (df['day'] >= 25).astype(int)
        
        # 季節
        df['season'] = df['month'].map({
            12: 'winter', 1: 'winter', 2: 'winter',
            3: 'spring', 4: 'spring', 5: 'spring', 
            6: 'summer', 7: 'summer', 8: 'summer',
            9: 'autumn', 10: 'autumn', 11: 'autumn'
        })
        
        # 季節編碼
        season_mapping = {'spring': 0, 'summer': 1, 'autumn': 2, 'winter': 3}
        df['season_encoded'] = df['season'].map(season_mapping)
        
        return df
    
    def add_rush_hour_features(self, df):
        """添加尖峰時段特徵"""
        df = df.copy()
        
        # 基礎尖峰時段
        df['is_morning_rush'] = df['hour'].between(
            self.morning_rush[0], self.morning_rush[1]-1).astype(int)
        df['is_evening_rush'] = df['hour'].between(
            self.evening_rush[0], self.evening_rush[1]-1).astype(int)
        df['is_noon_rush'] = df['hour'].between(
            self.noon_rush[0], self.noon_rush[1]-1).astype(int)
        
        # 綜合尖峰時段
        df['is_any_rush'] = (
            df['is_morning_rush'] | df['is_evening_rush'] | df['is_noon_rush']
        ).astype(int)
        
        # 深夜時段 (0-6點)
        df['is_night'] = df['hour'].between(0, 6).astype(int)
        
        # 工作時間 (9-17點)
        df['is_work_hours'] = df['hour'].between(9, 17).astype(int)
        
        # 尖峰強度 (數值化尖峰程度)
        df['rush_intensity'] = 0
        df.loc[df['is_morning_rush'] == 1, 'rush_intensity'] = 2  # 早尖峰最重要
        df.loc[df['is_evening_rush'] == 1, 'rush_intensity'] = 2  # 晚尖峰最重要
        df.loc[df['is_noon_rush'] == 1, 'rush_intensity'] = 1     # 午尖峰次要
        
        return df
    
    def add_weather_proxy_features(self, df, time_col='time_bin'):
        """添加氣象代理特徵 (基於時間推估)"""
        df = df.copy()
        
        # 雨季指標 (台灣梅雨季: 5-6月, 颱風季: 7-9月)
        df['is_rainy_season'] = df['month'].isin([5, 6, 7, 8, 9]).astype(int)
        
        # 寒流期 (12-2月)
        df['is_cold_season'] = df['month'].isin([12, 1, 2]).astype(int)
        
        # 學期期間 (排除寒暑假)
        df['is_school_term'] = (~df['month'].isin([7, 8, 1, 2])).astype(int)
        
        return df
    
    def add_cyclical_features(self, df):
        """添加週期性特徵 (sin/cos 編碼)"""
        df = df.copy()
        
        # 小時週期性
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        # 星期週期性
        df['weekday_sin'] = np.sin(2 * np.pi * df['weekday'] / 7)
        df['weekday_cos'] = np.cos(2 * np.pi * df['weekday'] / 7)
        
        # 月份週期性
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # 年內天數週期性
        df['dayofyear_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
        df['dayofyear_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
        
        return df
    
    def add_lag_features(self, df, speed_col='avg_speed', vehicle_col='total_vehicles'):
        """添加滯後特徵"""
        df = df.copy()
        df = df.sort_values('time_bin').reset_index(drop=True)
        
        # 速度滯後特徵
        for lag in [1, 2, 3, 6, 12, 24]:  # 1-3小時, 6小時, 12小時, 24小時前
            df[f'{speed_col}_lag_{lag}h'] = df[speed_col].shift(lag)
            if vehicle_col and vehicle_col in df.columns:
                df[f'{vehicle_col}_lag_{lag}h'] = df[vehicle_col].shift(lag)
        
        # 移動平均特徵
        for window in [3, 6, 12, 24]:  # 3, 6, 12, 24小時移動平均
            df[f'{speed_col}_ma_{window}h'] = df[speed_col].rolling(window=window).mean()
            if vehicle_col and vehicle_col in df.columns:
                df[f'{vehicle_col}_ma_{window}h'] = df[vehicle_col].rolling(window=window).mean()
        
        # 移動標準差 (變異性指標)
        for window in [6, 12, 24]:
            df[f'{speed_col}_std_{window}h'] = df[speed_col].rolling(window=window).std()
        
        return df
    
    def add_interaction_features(self, df):
        """添加交互特徵"""
        df = df.copy()
        
        # 週末 × 尖峰時段
        df['weekend_rush'] = df['is_weekend'] * df['is_any_rush']
        
        # 節假日 × 尖峰時段
        if 'is_holiday' in df.columns:
            df['holiday_rush'] = df['is_holiday'] * df['is_any_rush']
        
        # 雨季 × 尖峰時段
        df['rainy_rush'] = df['is_rainy_season'] * df['is_any_rush']
        
        # 季節 × 尖峰時段
        df['season_rush'] = df['season_encoded'] * df['rush_intensity']
        
        return df
    
    def create_all_features(self, df, time_col='time_bin', speed_col='avg_speed', vehicle_col='total_vehicles'):
        """建立所有特徵"""
        print("🔧 開始增強特徵工程...")
        
        # 確保時間欄位是datetime類型
        df[time_col] = pd.to_datetime(df[time_col])
        
        # 依序添加各種特徵
        df = self.add_time_features(df, time_col)
        df = self.add_calendar_features(df, time_col)
        df = self.add_rush_hour_features(df)
        df = self.add_weather_proxy_features(df, time_col)
        df = self.add_cyclical_features(df)
        df = self.add_lag_features(df, speed_col, vehicle_col)
        df = self.add_interaction_features(df)
        
        # 移除包含NaN的列 (由lag特徵產生)
        original_len = len(df)
        df = df.dropna()
        print(f"📊 特徵工程完成: {original_len} -> {len(df)} 筆資料")
        
        # 顯示新增的特徵
        original_cols = [time_col, speed_col, vehicle_col, 'vd_id']
        feature_cols = [col for col in df.columns if col not in original_cols]
        print(f"🚀 新增 {len(feature_cols)} 個特徵:")
        for i, col in enumerate(feature_cols, 1):
            if i <= 10:  # 只顯示前10個
                print(f"  {i}. {col}")
            elif i == 11:
                print(f"  ... 還有 {len(feature_cols)-10} 個特徵")
                break
        
        return df

def main():
    """測試特徵工程"""
    # 载入一個樣本資料進行測試
    from config import Config
    
    config = Config()
    parquet_files = glob.glob(os.path.join(config.CLEANED_DIR, "*.parquet"))
    
    if parquet_files:
        sample_file = parquet_files[0]
        print(f"📂 測試檔案: {os.path.basename(sample_file)}")
        
        df = pd.read_parquet(sample_file)
        print(f"📊 原始資料: {len(df)} 筆, {len(df.columns)} 個欄位")
        print(f"📋 原始欄位: {list(df.columns)}")
        
        # 應用特徵工程
        engineer = EnhancedFeatureEngineer()
        
        # 檢查欄位名稱
        speed_col = None
        vehicle_col = None
        time_col = 'time_bin'
        
        # 尋找速度欄位
        for col in df.columns:
            if 'speed' in col.lower():
                speed_col = col
                break
        
        # 尋找車流量欄位  
        for col in df.columns:
            if 'vehicle' in col.lower() or 'flow' in col.lower():
                vehicle_col = col
                break
        
        if not speed_col:
            print("❌ 找不到速度欄位")
            return
        
        print(f"🎯 使用欄位 - 時間: {time_col}, 速度: {speed_col}, 車流量: {vehicle_col}")
        
        df_enhanced = engineer.create_all_features(df, time_col, speed_col, vehicle_col)
        
        print(f"📈 增強後資料: {len(df_enhanced)} 筆, {len(df_enhanced.columns)} 個欄位")
        
        # 儲存範例
        output_path = os.path.join(config.BASE_DIR, "enhanced_features_sample.csv")
        df_enhanced.head(100).to_csv(output_path, index=False)
        print(f"💾 範例已儲存: {output_path}")
        
        # 顯示特徵統計
        print(f"\n📊 特徵統計:")
        numeric_cols = df_enhanced.select_dtypes(include=[np.number]).columns
        print(f"  數值特徵: {len(numeric_cols)} 個")
        print(f"  總特徵數: {len(df_enhanced.columns)} 個")
        
    else:
        print("❌ 找不到 cleaned 資料檔案")

if __name__ == "__main__":
    main()