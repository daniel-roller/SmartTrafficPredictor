# config.py

# ============================
# 資料設定
# ============================
VD_ID = "VD-N1-N-0.000-M-LOOP"

# 時間視窗設定
WINDOW = 12          # 用多少筆資料當輸入 (12 → 3 小時)
HORIZON = 96

# 特徵與目標欄位（含時間特徵）
FEATURES = [
    "avg_speed", "avg_occupancy", "total_vehicles",
    "hour", "weekday", "is_weekend", "is_peak"
]
TARGETS = ["avg_speed", "total_vehicles"]

# 資料分割比例
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15

# ============================
# 模型設定
# ============================
MODEL_TYPE   = "LSTM"   # 可選: LSTM, GRU, LSTM2, CNN_LSTM
LSTM_UNITS   = 64       # LSTM/GRU 隱藏層單元數
DROPOUT      = 0.2      # Dropout 機率
LEARNING_RATE = 1e-3    # 學習率
BATCH_SIZE   = 32       # 每次訓練批次大小
EPOCHS       = 100      # 最大訓練週期
PATIENCE     = 10       # EarlyStopping 容忍次數

# ============================
# Baseline 設定
# ============================
BASELINE_MA_WINDOW = 12  # Moving Average 的視窗大小

# ============================
# 圖片設定
# ============================
PLOT_SAMPLES = 200       # 畫圖時要顯示的資料筆數
    