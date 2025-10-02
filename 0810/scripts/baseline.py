# baseline.py
import os
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 匯入 baseline 的設定
from config import BASELINE_MA_WINDOW

# === 自動偵測最新的 dataset ===
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASET_DIR = os.path.join(BASE_DIR, "datasets")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

pkl_files = sorted(glob.glob(os.path.join(DATASET_DIR, "traffic_lstm_*.pkl")))
if not pkl_files:
    raise FileNotFoundError("❌ 在 datasets/ 找不到任何 .pkl，請先執行 make_dataset.py")
DATASET_PATH = pkl_files[-1]
print(f"✅ 使用的 dataset: {DATASET_PATH}")

# === 載入資料 ===
with open(DATASET_PATH, "rb") as f:
    data = pickle.load(f)

X_test, y_test = data["X_test"], data["y_test"]
y_mean, y_std = data["y_mean"], data["y_std"]
target_names = data["target_names"]

# 反標準化
y_true = y_test * y_std + y_mean

# === Baseline 1: Naive (上一筆值) ===
y_pred_naive = y_true[:-1]
y_true_naive = y_true[1:]

rmse_naive = np.sqrt(mean_squared_error(y_true_naive, y_pred_naive))
mae_naive = mean_absolute_error(y_true_naive, y_pred_naive)
mape_naive = np.mean(np.abs((y_true_naive - y_pred_naive) / (y_true_naive + 1e-8))) * 100

# === Baseline 2: Moving Average (使用 config 設定的 window) ===
window = BASELINE_MA_WINDOW
y_pred_ma = []
for i in range(window, len(y_true)):
    y_pred_ma.append(y_true[i - window : i].mean(axis=0))
y_pred_ma = np.array(y_pred_ma)
y_true_ma = y_true[window:]

rmse_ma = np.sqrt(mean_squared_error(y_true_ma, y_pred_ma))
mae_ma = mean_absolute_error(y_true_ma, y_pred_ma)
mape_ma = np.mean(np.abs((y_true_ma - y_pred_ma) / (y_true_ma + 1e-8))) * 100

# === 儲存結果 ===
with open(os.path.join(RESULTS_DIR, "baseline_metrics.txt"), "w", encoding="utf-8") as f:
    f.write("Naive Baseline:\n")
    f.write(f"  RMSE: {rmse_naive:.4f}, MAE: {mae_naive:.4f}, MAPE: {mape_naive:.2f}%\n\n")
    f.write(f"Moving Average ({window}) Baseline:\n")
    f.write(f"  RMSE: {rmse_ma:.4f}, MAE: {mae_ma:.4f}, MAPE: {mape_ma:.2f}%\n")

print("✅ Baseline 指標已存到 baseline_metrics.txt")

# === 畫圖 (只畫車速，前200筆) ===
plt.figure(figsize=(12, 6))
plt.plot(y_true[:200, 0], label=f"True {target_names[0]}")
plt.plot(y_pred_naive[:200, 0], label=f"Naive Pred {target_names[0]}")
plt.plot(y_pred_ma[:200, 0], label=f"MA({window}) Pred {target_names[0]}")
plt.title(f"{target_names[0]} Baseline (前200筆)")
plt.xlabel("Samples")
plt.ylabel(target_names[0])
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(RESULTS_DIR, "baseline_pred_vs_true.png"))
plt.close()

print("✅ 圖片已存到 baseline_pred_vs_true.png")
