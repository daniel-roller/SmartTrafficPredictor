# -*- coding: utf-8 -*-
"""
配置文件 - 完整修正版 (只使用 datasets 資料夾)
"""

import os
import numpy as np
from datetime import datetime
import glob

class Config:
    """系統配置類"""
    
    def __init__(self):
        # 基礎路徑 - 當前工作目錄
        self.BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        
        # 實驗時間戳
        self.EXPERIMENT_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 結果根目錄 - 在當前目錄下建立 results 資料夾
        self.RESULTS_ROOT = os.path.join(self.BASE_DIR, "results")
        
        # 當前實驗目錄 - 每次執行建立新的實驗資料夾
        self.EXPERIMENT_DIR = os.path.join(self.RESULTS_ROOT, f"experiment_{self.EXPERIMENT_TIMESTAMP}")
        
        # 各種輸出目錄 - 都在實驗目錄下
        self.PLOTS_DIR = os.path.join(self.EXPERIMENT_DIR, "plots")
        self.MODELS_DIR = os.path.join(self.EXPERIMENT_DIR, "models") 
        self.LOGS_DIR = os.path.join(self.EXPERIMENT_DIR, "logs")
        self.RESULTS_DIR = self.EXPERIMENT_DIR  # 報告直接放在實驗目錄
        
        # 數據目錄 - 只使用 datasets 資料夾
        self.DATA_DIR = os.path.join(self.BASE_DIR, "datasets")  # 主要數據來源
        self.DATASETS_DIR = os.path.join(self.BASE_DIR, "datasets")  # .npy 檔案
        
        # 系統參數
        self.RANDOM_STATE = 42       # 隨機種子
        self.N_JOBS = -1            # 並行處理核心數 (-1 = 使用所有核心)
        self.VERBOSE = 1            # 詳細輸出等級
        
        # 數據分割比例
        self.TRAIN_RATIO = 0.7      # 訓練集比例
        self.VAL_RATIO = 0.15       # 驗證集比例  
        self.TEST_RATIO = 0.15      # 測試集比例
        
        # 數據處理參數
        self.MAX_SAMPLE_SIZE = 10000  # 最大樣本數量
        self.SCALER_METHOD = 'standard'  # 數據標準化方法: 'standard', 'minmax', 'robust'
        self.CV_FOLDS = 5  # 交叉驗證折數
        
        # 特徵工程參數
        self.USE_LAG_FEATURES = True        # 是否使用滯後特徵
        self.LAG_WINDOWS = [1, 2, 3, 6, 12, 24]  # 滯後窗口
        self.LAG_FEATURES = [1, 2, 3]       # 簡化版滯後特徵 (向後相容)
        self.USE_ROLLING_FEATURES = True    # 是否使用滾動統計特徵
        self.ROLLING_WINDOWS = [3, 6, 12, 24]  # 滾動窗口
        self.USE_TIME_FEATURES = True       # 是否使用時間特徵
        
        # 確保所有目錄存在
        self._create_directories()
        
        # 自動檢測可用的數據檔案
        self.available_datasets = self._detect_datasets()
        
        # 設定 DATASETS_INFO (向後相容)
        self.DATASETS_INFO = self.available_datasets
        
        # 傳統機器學習模型參數 - 簡化為3個基本模型
        self.TRADITIONAL_ML_PARAMS = {
            'Ridge': {
                'alpha': 1.0,
                'solver': 'auto',
                'random_state': self.RANDOM_STATE
            },
            'RandomForest': {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': self.RANDOM_STATE,
                'n_jobs': self.N_JOBS
            },
            'XGBoost': {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': self.RANDOM_STATE,
                'n_jobs': self.N_JOBS
            }
        }
        
        # 是否儲存模型
        self.SAVE_MODELS = True
        
        # 模型配置 (向後相容)
        self.MODELS_CONFIG = self.TRADITIONAL_ML_PARAMS
        
        # 數據處理配置
        self.FEATURE_CONFIG = {
            'time_features': self.USE_TIME_FEATURES,
            'lag_features': self.LAG_WINDOWS,
            'rolling_features': self.ROLLING_WINDOWS,
            'weather_features': True,
            'scaler_method': self.SCALER_METHOD
        }
        
        # 評估配置
        self.EVALUATION_CONFIG = {
            'test_size': 0.2,
            'cv_folds': self.CV_FOLDS,  # 使用統一的 CV_FOLDS
            'random_state': self.RANDOM_STATE,
            'scoring': ['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error']
        }
        
        # 可視化配置
        self.PLOT_CONFIG = {
            'figsize': (12, 8),
            'dpi': 300,
            'style': 'whitegrid',
            'color_palette': 'husl'
        }
        
        # 訓練配置
        self.TRAINING_CONFIG = {
            'batch_size': 64,
            'epochs': 100,
            'patience': 10,
            'validation_split': 0.2,
            'shuffle': True,
            'random_state': self.RANDOM_STATE
        }
        
        # 日誌配置
        self.LOGGING_CONFIG = {
            'level': 'INFO',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'file_handler': True,
            'console_handler': True
        }
        
        # 實驗配置
        self.EXPERIMENT_CONFIG = {
            'save_models': True,
            'save_predictions': True,
            'generate_plots': True,
            'detailed_logs': True
        }
    
    def _create_directories(self):
        """建立所有必要的目錄"""
        directories = [
            self.RESULTS_ROOT,
            self.EXPERIMENT_DIR,
            self.PLOTS_DIR,
            self.MODELS_DIR,
            self.LOGS_DIR,
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            
        print(f"📁 實驗目錄已建立: {self.EXPERIMENT_DIR}")
    
    def create_directories(self):
        """向後相容的方法名稱"""
        return self._create_directories()
    
    def _detect_datasets(self):
        """自動檢測可用的 .npy 數據檔案"""
        datasets = {}
        
        print("🔍 掃描 datasets 資料夾中的 .npy 檔案...")
        
        # 只檢查 .npy 檔案 (在 datasets 資料夾)
        if os.path.exists(self.DATASETS_DIR):
            npy_files = glob.glob(os.path.join(self.DATASETS_DIR, "*.npy"))
            
            if npy_files:
                print(f"📦 在 {self.DATASETS_DIR} 找到 {len(npy_files)} 個 .npy 檔案:")
                
                for npy_file in npy_files:
                    name = os.path.splitext(os.path.basename(npy_file))[0]
                    file_size = os.path.getsize(npy_file) / (1024 * 1024)  # MB
                    
                    datasets[name] = {
                        'type': 'npy',
                        'path': npy_file,
                        'source': 'datasets',
                        'size_mb': round(file_size, 2)
                    }
                    print(f"  📄 {name} ({file_size:.2f} MB)")
                
                print(f"✅ 總計找到 {len(datasets)} 個可用數據集")
            else:
                print(f"❌ 在 {self.DATASETS_DIR} 中未找到任何 .npy 檔案")
        else:
            print(f"❌ datasets 資料夾不存在: {self.DATASETS_DIR}")
            
        if not datasets:
            print("⚠️ 未找到任何數據檔案，將使用模擬數據")
            
        return datasets
    
    def get_dataset_info(self, dataset_name: str = None):
        """獲取數據集資訊"""
        if dataset_name:
            return self.available_datasets.get(dataset_name)
        else:
            return self.available_datasets
    
    def load_dataset(self, dataset_name: str):
        """載入指定數據集"""
        dataset_info = self.get_dataset_info(dataset_name)
        
        if not dataset_info:
            raise FileNotFoundError(f"數據集 '{dataset_name}' 不存在")
        
        file_path = dataset_info['path']
        
        try:
            print(f"📊 載入數據集: {dataset_name}")
            print(f"   📁 檔案路徑: {file_path}")
            print(f"   💾 檔案大小: {dataset_info['size_mb']} MB")
            
            # 載入 .npy 檔案
            data = np.load(file_path, allow_pickle=True)
            
            # 檢查數據結構
            if isinstance(data, dict):
                print(f"   📊 數據類型: 字典 (包含 {len(data)} 個鍵)")
                for key in list(data.keys())[:5]:  # 只顯示前5個鍵
                    print(f"      🔑 {key}: {type(data[key])}")
                if len(data) > 5:
                    print(f"      ... 還有 {len(data) - 5} 個鍵")
            elif isinstance(data, np.ndarray):
                print(f"   📊 數據類型: NumPy陣列")
                print(f"   📏 數據形狀: {data.shape}")
                print(f"   🎯 數據型別: {data.dtype}")
            else:
                print(f"   📊 數據類型: {type(data)}")
            
            print(f"✅ 數據載入成功!")
            return data
            
        except Exception as e:
            print(f"❌ 載入數據失敗: {e}")
            return None
    
    def get_dataset_list(self):
        """獲取所有可用數據集名稱列表"""
        return list(self.available_datasets.keys())
    
    def get_data_files(self):
        """獲取數據檔案路徑列表 (向後相容)"""
        return [info['path'] for info in self.available_datasets.values()]
    
    def print_config(self):
        """印出當前配置"""
        print("🔧 系統配置資訊:")
        print(f"   📂 基礎目錄: {self.BASE_DIR}")
        print(f"   📊 實驗目錄: {self.EXPERIMENT_DIR}")
        print(f"   🖼️  圖片目錄: {self.PLOTS_DIR}")
        print(f"   📄 結果目錄: {self.RESULTS_DIR}")
        print(f"   📦 數據目錄: {self.DATASETS_DIR}")
        print(f"   🧪 實驗ID: experiment_{self.EXPERIMENT_TIMESTAMP}")
        print(f"   🎲 隨機種子: {self.RANDOM_STATE}")
        print(f"   📊 數據分割: 訓練{self.TRAIN_RATIO*100:.0f}% / 驗證{self.VAL_RATIO*100:.0f}% / 測試{self.TEST_RATIO*100:.0f}%")
        print(f"   🔄 標準化方法: {self.SCALER_METHOD}")
        print(f"   🔢 交叉驗證: {self.CV_FOLDS} 折")
        
        print(f"\n📊 可用數據集 ({len(self.available_datasets)} 個):")
        if self.available_datasets:
            for name, info in self.available_datasets.items():
                print(f"   📦 {name} ({info['size_mb']} MB)")
        else:
            print("   ❌ 未找到數據檔案")
        
        print(f"\n🤖 傳統ML模型 ({len(self.TRADITIONAL_ML_PARAMS)} 個):")
        for model_name in self.TRADITIONAL_ML_PARAMS.keys():
            print(f"   🔧 {model_name}")
        
        print(f"\n📋 重要屬性狀態:")
        print(f"   DATASETS_INFO: {'✅' if hasattr(self, 'DATASETS_INFO') else '❌'}")
        print(f"   TRADITIONAL_ML_PARAMS: {'✅' if hasattr(self, 'TRADITIONAL_ML_PARAMS') else '❌'}")
        print(f"   MODELS_CONFIG: {'✅' if hasattr(self, 'MODELS_CONFIG') else '❌'}")
        print(f"   CV_FOLDS: {'✅' if hasattr(self, 'CV_FOLDS') else '❌'}")
    
    def check_data_availability(self):
        """檢查數據可用性"""
        print("🔍 數據檔案檢查:")
        
        if os.path.exists(self.DATASETS_DIR):
            files = os.listdir(self.DATASETS_DIR)
            npy_files = [f for f in files if f.endswith('.npy')]
            
            if npy_files:
                total_size = 0
                print(f"   ✅ datasets 資料夾: {len(npy_files)} 個 .npy 檔案")
                
                for file in npy_files:
                    file_path = os.path.join(self.DATASETS_DIR, file)
                    file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
                    total_size += file_size
                    print(f"      📄 {file} ({file_size:.2f} MB)")
                
                print(f"   💾 總大小: {total_size:.2f} MB")
                return True
            else:
                print(f"   📁 datasets 資料夾存在但無 .npy 檔案")
                return False
        else:
            print(f"   ❌ datasets 資料夾不存在: {self.DATASETS_DIR}")
            return False
    
    def get_random_state(self):
        """獲取隨機種子"""
        return self.RANDOM_STATE
    
    def set_random_state(self, seed: int):
        """設定隨機種子"""
        self.RANDOM_STATE = seed
        # 同時更新模型配置中的隨機種子
        for model_config in self.TRADITIONAL_ML_PARAMS.values():
            if 'random_state' in model_config:
                model_config['random_state'] = seed
        for model_config in self.MODELS_CONFIG.values():
            if 'random_state' in model_config:
                model_config['random_state'] = seed
        self.EVALUATION_CONFIG['random_state'] = seed
        self.TRAINING_CONFIG['random_state'] = seed
        print(f"🎲 隨機種子已設定為: {seed}")
    
    def get_cv_folds(self):
        """獲取交叉驗證折數"""
        return self.CV_FOLDS
    
    def set_cv_folds(self, folds: int):
        """設定交叉驗證折數"""
        if folds < 2:
            raise ValueError("交叉驗證折數必須至少為 2")
        
        self.CV_FOLDS = folds
        self.EVALUATION_CONFIG['cv_folds'] = folds
        print(f"🔢 交叉驗證折數已設定為: {folds}")
    
    def get_scaler_method(self):
        """獲取標準化方法"""
        return self.SCALER_METHOD
    
    def set_scaler_method(self, method: str):
        """設定標準化方法"""
        valid_methods = ['standard', 'minmax', 'robust']
        if method not in valid_methods:
            raise ValueError(f"無效的標準化方法: {method}. 可用方法: {valid_methods}")
        
        self.SCALER_METHOD = method
        self.FEATURE_CONFIG['scaler_method'] = method
        print(f"🔄 標準化方法已設定為: {method}")
    
    def get_available_datasets_summary(self):
        """獲取數據集摘要資訊"""
        summary = {
            'total_count': len(self.available_datasets),
            'total_size_mb': sum(info['size_mb'] for info in self.available_datasets.values()),
            'datasets': self.available_datasets
        }
        return summary
    
    def validate_dataset_exists(self, dataset_name: str) -> bool:
        """驗證數據集是否存在"""
        exists = dataset_name in self.available_datasets
        if not exists:
            available = list(self.available_datasets.keys())
            print(f"❌ 數據集 '{dataset_name}' 不存在")
            print(f"📋 可用數據集: {available}")
        return exists
    
    def get_feature_engineering_config(self):
        """獲取特徵工程配置"""
        return {
            'use_lag_features': self.USE_LAG_FEATURES,
            'lag_windows': self.LAG_WINDOWS,
            'use_rolling_features': self.USE_ROLLING_FEATURES,
            'rolling_windows': self.ROLLING_WINDOWS,
            'use_time_features': self.USE_TIME_FEATURES,
            'scaler_method': self.SCALER_METHOD
        }
    
    def get_traditional_ml_params(self):
        """獲取傳統ML模型參數"""
        return self.TRADITIONAL_ML_PARAMS
    
    def refresh_datasets_info(self):
        """重新掃描並更新數據集資訊"""
        print("🔄 重新掃描數據集...")
        self.available_datasets = self._detect_datasets()
        self.DATASETS_INFO = self.available_datasets
        print(f"✅ 數據集資訊已更新: {len(self.DATASETS_INFO)} 個數據集")
        return self.DATASETS_INFO
    
    def get_datasets_info_legacy(self):
        """獲取 DATASETS_INFO (向後相容方法)"""
        return self.DATASETS_INFO

# 建立全域配置實例
config = Config()

# 如果直接執行此檔案，顯示配置資訊
if __name__ == "__main__":
    config.print_config()
    print("\n" + "="*50)
    has_data = config.check_data_availability()
    
    if has_data:
        print("\n" + "="*50)
        print("🧪 測試數據載入:")
        
        # 測試載入第一個數據集
        dataset_names = config.get_dataset_list()
        if dataset_names:
            test_dataset = dataset_names[0]
            print(f"\n📊 測試載入: {test_dataset}")
            data = config.load_dataset(test_dataset)
            
            if data is not None:
                print("✅ 數據載入測試成功!")
            else:
                print("❌ 數據載入測試失敗!")
    
    # 測試重要屬性
    print(f"\n🔍 屬性檢查:")
    print(f"   DATASETS_INFO: {list(config.DATASETS_INFO.keys())}")
    print(f"   TRADITIONAL_ML_PARAMS: {list(config.TRADITIONAL_ML_PARAMS.keys())}")
    print(f"   CV_FOLDS: {config.CV_FOLDS}")