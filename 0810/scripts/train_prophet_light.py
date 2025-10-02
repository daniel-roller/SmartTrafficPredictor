# train_prophet_light.py
import os
import glob
import pickle
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

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

# === 載入 cleaned 資料 ===
CLEANED_DIR = os.path.join(BASE_DIR, "cleaned")
files = sorted(glob.glob(os.path.join(CLEANED_DIR, "*.parquet")))
if not files:
    raise FileNotFoundError("❌ 找不到 parquet 檔案")

# 只讀取前 30 天，避免一次載入太大
dfs = []
for f in files[:30]:
    dfs.append(pd.read_parquet(f))
df_all = pd.concat(dfs, ignore_index=True)

# 移除 timezone
df_all["time_bin"] = pd.to_datetime(df_all["time_bin"]).dt.tz_localize(None)

# === 聚合成「每小時平均」，只針對數值欄位 ===
df_hour = (
    df_all.set_index("time_bin")
    .select_dtypes(include=[np.number])  # 只取數值欄位
    .resample("1h")                      # 每小時
    .mean()
    .reset_index()
)

# 用 avg_speed 做 Prophet 測試
df_prophet = df_hour[["time_bin", "avg_speed"]].dropna().rename(
    columns={"time_bin": "ds", "avg_speed": "y"}
)

print(f"✅ 測試資料大小: {df_prophet.shape}")

# === 建立 Prophet 模型 ===
m = Prophet(
    yearly_seasonality=False,
    weekly_seasonality=True,
    daily_seasonality=True,
    seasonality_mode="additive"
)
m.fit(df_prophet)

# === 預測未來 7 天（小時為單位） ===
future = m.make_future_dataframe(periods=7*24, freq="h")  # 7 天 * 24 小時
forecast = m.predict(future)

# === 評估 (只在真實值存在的部分) ===
df_merge = pd.merge(df_prophet, forecast[["ds", "yhat"]], on="ds", how="inner")
mae = mean_absolute_error(df_merge["y"], df_merge["yhat"])
rmse = np.sqrt(mean_squared_error(df_merge["y"], df_merge["yhat"]))

with open(os.path.join(RESULTS_DIR, "metrics_prophet_light.txt"), "w", encoding="utf-8") as f:
    f.write(f"Prophet (light) MAE: {mae:.4f}\n")
    f.write(f"Prophet (light) RMSE: {rmse:.4f}\n")

print("✅ Prophet (light) 結果已存到 metrics_prophet_light.txt")

# === 繪圖 ===
fig = m.plot(forecast)
plt.title("Prophet 測試預測 (未來7天, 小時粒度)")
plt.savefig(os.path.join(RESULTS_DIR, "prophet_forecast_light.png"))
plt.close()
print("✅ 預測圖已存到 prophet_forecast_light.png")
