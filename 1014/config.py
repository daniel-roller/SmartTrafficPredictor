# -*- coding: utf-8 -*-
"""
交通流量預測專案 - 統一設定檔 (SVM優化版)
"""

import os
import multiprocessing
from datetime import datetime

class Config:
    """專案配置類"""
    
    def __init__(self):
        # 系統資源配置
        self.CPU_COUNT = multiprocessing.cpu_count()
        self.MAX_WORKERS_DATASETS = min(self.CPU_COUNT, 3)
        self.MAX_WORKERS_MODELS = min(3, self.CPU_COUNT)
        
        print(f"🔧 系統資源: {self.CPU_COUNT} 核心 CPU")
        print(f"🔧 資料集執行緒: {self.MAX_WORKERS_DATASETS}")
        print(f"🔧 模型訓練執行緒: {self.MAX_WORKERS_MODELS}")
        
        # 基礎路徑
        self.BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        self.PROJECT_ROOT = self.BASE_DIR
        
        # 資料路徑
        self.DATA_DIR = os.path.join(self.BASE_DIR, "select")
        self.MODELS_DIR = os.path.join(self.BASE_DIR, "models")
        self.RESULTS_DIR = os.path.join(self.BASE_DIR, "results")
        self.LOGS_DIR = os.path.join(self.BASE_DIR, "logs")
        
        # 建立目錄
        for directory in [self.MODELS_DIR, self.RESULTS_DIR, self.LOGS_DIR]:
            os.makedirs(directory, exist_ok=True)
            os.makedirs(os.path.join(directory, "plots"), exist_ok=True)
            os.makedirs(os.path.join(directory, "metrics"), exist_ok=True)
            os.makedirs(os.path.join(directory, "predictions"), exist_ok=True)
            os.makedirs(os.path.join(directory, "analysis"), exist_ok=True)
            
        # 為每個模型建立子目錄
        for model_name in ['xgboost', 'randomforest', 'svm']:
            os.makedirs(os.path.join(self.MODELS_DIR, model_name), exist_ok=True)
        
        # 驗證資料目錄是否存在
        if not os.path.exists(self.DATA_DIR):
            print(f"❌ 資料目錄不存在: {self.DATA_DIR}")
        else:
            print(f"✅ 資料目錄: {self.DATA_DIR}")
            csv_files = [f for f in os.listdir(self.DATA_DIR) if f.endswith('.csv')]
            print(f"📁 找到 {len(csv_files)} 個 CSV 檔案")
            for csv_file in csv_files:
                print(f"   📄 {csv_file}")
        
        # 優化版模型參數
        self.MODELS = {
            'XGBoost': {
                'n_estimators': 50,        # 增加樹的數量
                'max_depth': 8,            # 增加深度
                'learning_rate': 0.08,     # 降低學習率提升精度
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'n_jobs': 1,
                'verbosity': 0
            },
            'RandomForest': {
                'n_estimators': 50,        # 增加樹的數量
                'max_depth': 12,           # 增加深度
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'max_features': 'sqrt',
                'random_state': 42,
                'n_jobs': 1,
                'verbose': 0
            },
            'SVM': {
                'C': 1000.0,               # 大幅增加C值
                'gamma': 'scale',          # 使用scale自動調整
                'kernel': 'rbf',
                'epsilon': 0.1,            # 增加容忍度
                'cache_size': 1000,        # 增加快取
                'max_iter': 10000          # 增加最大迭代次數
            }
        }
        
        # 時間設定
        self.PEAK_HOURS = [(7, 9), (17, 19)]
        self.HOLIDAYS = ['2024-01-01', '2024-12-25']
        
        # 特徵工程參數
        self.LAG_FEATURES = [1, 2, 6]
        self.ROLLING_WINDOWS = [3, 6]
        
        # 評估指標
        self.METRICS = ['MAE', 'RMSE', 'R2', 'MAPE']
        
        # 資料分割
        self.TRAIN_SIZE = 0.2              # 增加到20%
        self.VAL_SIZE = 0.05
        self.TEST_SIZE = 0.05
        
        # 資料採樣設定
        self.SAMPLE_SIZE = 20000           # 增加到20000筆
        
        # 隨機種子
        self.RANDOM_STATE = 42
        
        # 時間戳
        self.TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 多執行緒設定
        self.THREAD_TIMEOUT = 600          # 10分鐘超時
        self.ENABLE_PROGRESS_BAR = True
        
        # 記憶體管理
        self.MAX_MEMORY_USAGE = 0.8
        
        # 模型比較設定
        self.COMPARISON_METRICS = ['R2', 'RMSE', 'MAE', 'MAPE']
        self.SAVE_PLOTS = True
        self.SHOW_PLOTS = False

config = Config()