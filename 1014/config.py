import os
import multiprocessing

class Config:
    # === 資料夾路徑 ===
    # 修正：取得當前檔案的目錄作為基礎目錄
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 這會指向 1014/
    
    RAW_DIR = os.path.join(BASE_DIR, "raw")
    CLEANED_DIR = os.path.join(BASE_DIR, "cleaned")
    DATASETS_DIR = os.path.join(BASE_DIR, "datasets")
    MODELS_DIR = os.path.join(BASE_DIR, "models")
    RESULTS_DIR = os.path.join(BASE_DIR, "results")
    
    # 確保資料夾存在
    for dir_path in [DATASETS_DIR, MODELS_DIR, RESULTS_DIR]:
        os.makedirs(dir_path, exist_ok=True)
    
    # === 多執行緒設定 ===
    CPU_COUNT = multiprocessing.cpu_count()
    MAX_WORKERS_MODELS = min(3, CPU_COUNT)      
    MAX_WORKERS_SEGMENTS = min(2, CPU_COUNT)    
    
    # === 時間窗設定（簡化測試） ===
    WINDOW_SIZE = 24      # 固定1天歷史資料
    HORIZON = 24          # 固定預測1天
    
    # === 模型參數（簡化版） ===
    LSTM_PARAMS = {
        'units': 64,
        'dropout': 0.2,
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs': 20,  # 快速測試
        'patience': 5
    }
    
    TCN_PARAMS = {
        'nb_filters': 32,
        'kernel_size': 3,
        'dropout_rate': 0.2,
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs': 20,
        'patience': 5
    }
    
    TRANSFORMER_PARAMS = {
        'model_dim': 64,
        'num_heads': 4,
        'num_layers': 2,
        'dropout': 0.2,
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs': 20,
        'patience': 5
    }
    
    # === 特徵設定 ===
    FEATURES = ["avg_speed", "total_vehicles", "hour", "weekday", "is_weekend", "is_peak"]
    TARGETS = ["avg_speed", "total_vehicles"]
    
    # === 資料分割 ===
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15
    
    def __init__(self):
        print(f"🔧 Config 載入完成")
        print(f"📁 基礎目錄: {self.BASE_DIR}")
        print(f"🧹 清理資料: {self.CLEANED_DIR}")
        print(f"📦 資料集: {self.DATASETS_DIR}")
        print(f"🤖 模型: {self.MODELS_DIR}")
        print(f"📊 結果: {self.RESULTS_DIR}")
        print(f"💻 CPU: {self.CPU_COUNT}, 模型執行緒: {self.MAX_WORKERS_MODELS}")