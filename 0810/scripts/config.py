# config.py
# 統一設定檔：LSTM + Prophet

# ============================
# 資料設定
# ============================
VD_ID = "VD-N1-N-0.000-M-LOOP"

# 時間視窗設定（LSTM 用）
WINDOW = 24          # ⬅️ 改成 24（6 小時），可看更長趨勢
HORIZON = 96         # ⬅️ 只預測 1 天，短期模型先穩定

# 特徵與目標欄位
FEATURES = [
    "avg_speed", "avg_occupancy", "total_vehicles",
    "hour", "weekday", "is_weekend", "is_peak"
]
TARGETS = ["avg_speed", "total_vehicles"]

# 資料分割比例
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15

# ============================
# 模型設定（LSTM 系列）
# ============================
MODEL_TYPE   = "LSTM"
LSTM_UNITS   = 128       # ⬅️ 提升模型容量
DROPOUT      = 0.3
LEARNING_RATE = 1e-3
BATCH_SIZE   = 32
EPOCHS       = 100
PATIENCE     = 10

# ============================
# Baseline 設定
# ============================
BASELINE_MA_WINDOW = 12

# ============================
# Prophet 長期趨勢設定
# ============================
AGG_FREQ_LONG = "1D"
LONG_TARGET = "avg_speed"
LONG_HORIZON_DAYS = 30
USE_REGRESSORS = True
TREND_MODE = "flat"
CAP_MULTIPLIER = 1.1
FLOOR_MULTIPLIER = 0.9
SMOOTHNESS = 0.35   # ⬅️ 原本 0.3，改成 0.45 讓趨勢更靈活


# ============================
# 回測設定
# ============================
BACKTEST_HORIZON_DAYS = 30
BACKTEST_STEPS = 3
