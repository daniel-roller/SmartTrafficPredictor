# predict.py
import os
import glob
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow import keras

from config import MODEL_TYPE

# === 自動設定路徑 ===
BASE_DIR    = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASET_DIR = os.path.join(BASE_DIR, "datasets")
MODEL_DIR   = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# === 載入最新 dataset ===
pkl_files = sorted(glob.glob(os.path.join(DATASET_DIR, "traffic_lstm_*.pkl")))
if not pkl_files:
    raise FileNotFoundError("❌ 在 datasets/ 找不到任何 .pkl，請先執行 make_dataset.py")
DATASET_PATH = pkl_files[-1]
print(f"✅ 使用的 dataset: {DATASET_PATH}")

with open(DATASET_PATH, "rb") as f:
    data = pickle.load(f)

X_test, y_test = data["X_test"], data["y_test"]
FEATURES       = data["feature_names"]
TARGETS        = data["target_names"]
VD_ID          = data["vd_id"]
WINDOW         = data["window"]
HORIZON        = data["horizon"]
x_mean, x_std  = data["x_mean"], data["x_std"]
y_mean, y_std  = data["y_mean"], data["y_std"]

# 新增：取出測試集時間戳
test_time_bin = data.get("test_time_bin", None)

# === 載入模型 ===
model_path = os.path.join(MODEL_DIR, f"{MODEL_TYPE.lower()}_baseline.best.h5")
if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ 找不到模型檔案 {model_path}，請先執行 train_lstm.py")

model = keras.models.load_model(model_path)
print(f"✅ 載入模型: {model_path}")

# === 預測 ===
y_pred = model.predict(X_test)

# 反標準化
y_true = y_test * y_std + y_mean
y_pred = y_pred * y_std + y_mean

# === 整體誤差 ===
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae  = mean_absolute_error(y_true, y_pred)
mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

print(f"\n📊 測試集結果：RMSE={rmse:.2f}, MAE={mae:.2f}, MAPE={mape:.2f}%")

# === 尖峰 vs 離峰誤差 ===
if test_time_bin is not None:
    hours = pd.to_datetime(test_time_bin).hour
    # ✅ 改用向量化判斷，不會報錯
    is_peak = (((7 <= hours) & (hours <= 9)) | ((16 <= hours) & (hours <= 19))).astype(int)

    # 尖峰
    mae_peak = mean_absolute_error(y_true[is_peak==1,0], y_pred[is_peak==1,0])
    mape_peak = np.mean(np.abs((y_true[is_peak==1,0] - y_pred[is_peak==1,0]) / (y_true[is_peak==1,0] + 1e-8))) * 100

    # 離峰
    mae_off = mean_absolute_error(y_true[is_peak==0,0], y_pred[is_peak==0,0])
    mape_off = np.mean(np.abs((y_true[is_peak==0,0] - y_pred[is_peak==0,0]) / (y_true[is_peak==0,0] + 1e-8))) * 100

    print("\n--- 分情境誤差分析 ---")
    print(f"尖峰: MAE={mae_peak:.2f}, MAPE={mape_peak:.2f}%")
    print(f"離峰: MAE={mae_off:.2f}, MAPE={mape_off:.2f}%")

# === 存檔 ===
with open(os.path.join(RESULTS_DIR, "metrics_test.txt"), "w", encoding="utf-8") as f:
    f.write(f"Test RMSE: {rmse:.4f}\n")
    f.write(f"Test MAE: {mae:.4f}\n")
    f.write(f"Test MAPE: {mape:.4f}%\n")

    if test_time_bin is not None:
        f.write("\n--- 分情境誤差分析 ---\n")
        f.write(f"尖峰: MAE={mae_peak:.2f}, MAPE={mape_peak:.2f}%\n")
        f.write(f"離峰: MAE={mae_off:.2f}, MAPE={mape_off:.2f}%\n")

# === 畫圖 (速度 & 車流量) ===
plt.figure(figsize=(12,6))
plt.plot(y_true[:200,0], label="True Speed")
plt.plot(y_pred[:200,0], label="Predicted Speed")
plt.xlabel("Samples")
plt.ylabel("Speed (km/h)")
plt.title("Prediction vs True (Speed)")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(RESULTS_DIR, f"pred_vs_true_{MODEL_TYPE}.png"))
plt.close()

plt.figure(figsize=(12,6))
plt.plot(y_true[:200,1], label="True Flow")
plt.plot(y_pred[:200,1], label="Predicted Flow")
plt.xlabel("Samples")
plt.ylabel("Vehicles")
plt.title("Prediction vs True (Flow)")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(RESULTS_DIR, f"pred_vs_true_flow_{MODEL_TYPE}.png"))
plt.close()

print(f"\n✅ 結果已存到 {RESULTS_DIR}")
