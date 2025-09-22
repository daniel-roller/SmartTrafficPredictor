# predict.py
import os
import pickle
import glob
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error

# === 自動偵測最新的 dataset ===
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASET_DIR = os.path.join(BASE_DIR, "datasets")
MODEL_PATH = os.path.join(BASE_DIR, "models", "lstm_baseline.best.h5")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

pkl_files = sorted(glob.glob(os.path.join(DATASET_DIR, "traffic_lstm_*.pkl")))
if not pkl_files:
    raise FileNotFoundError("❌ 在 datasets/ 找不到任何 .pkl，請先執行 make_dataset.py")
DATASET_PATH = pkl_files[-1]  # 選最新的一個
print(f"✅ 使用的 dataset: {DATASET_PATH}")

# === 載入資料 ===
with open(DATASET_PATH, "rb") as f:
    data = pickle.load(f)

X_test, y_test = data["X_test"], data["y_test"]
y_mean, y_std = data["y_mean"], data["y_std"]
target_names = data["target_names"]

print("資料形狀：")
print("  X_test:", X_test.shape, "y_test:", y_test.shape)

# === 載入模型（不 compile，避免 keras.metrics.mse 問題） ===
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# === 預測 ===
y_pred_norm = model.predict(X_test)
y_pred = y_pred_norm * y_std + y_mean
y_true = y_test * y_std + y_mean

# === 指標 ===
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae = mean_absolute_error(y_true, y_pred)
mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

with open(os.path.join(RESULTS_DIR, "metrics_test.txt"), "w", encoding="utf-8") as f:
    f.write(f"Test RMSE: {rmse:.4f}\n")
    f.write(f"Test MAE: {mae:.4f}\n")
    f.write(f"Test MAPE: {mape:.2f}%\n")

print("✅ 測試集指標已存到 metrics_test.txt")

# === 畫圖 (車速 + 車流量) ===
plt.figure(figsize=(12, 8))

# 車速
plt.subplot(2, 1, 1)
plt.plot(y_true[:200, 0], label=f"True {target_names[0]}")
plt.plot(y_pred[:200, 0], label=f"Pred {target_names[0]}")
plt.title(f"{target_names[0]} 真實 vs 預測 (前200筆)")
plt.xlabel("Samples")
plt.ylabel(target_names[0])
plt.legend()
plt.grid(True)

# 車流量
plt.subplot(2, 1, 2)
plt.plot(y_true[:200, 1], label=f"True {target_names[1]}")
plt.plot(y_pred[:200, 1], label=f"Pred {target_names[1]}")
plt.title(f"{target_names[1]} 真實 vs 預測 (前200筆)")
plt.xlabel("Samples")
plt.ylabel(target_names[1])
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "pred_vs_true.png"))
plt.close()

print("✅ 圖片已存到 pred_vs_true.png")
