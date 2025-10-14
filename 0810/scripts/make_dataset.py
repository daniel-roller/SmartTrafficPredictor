# make_dataset.py
import os
import glob
import pickle
import numpy as np
import pandas as pd

from config import VD_ID, WINDOW, HORIZON, FEATURES, TARGETS, TRAIN_RATIO, VAL_RATIO

# === 自動設定資料夾路徑 ===
BASE_DIR    = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CLEANED_DIR = os.path.join(BASE_DIR, "cleaned")
OUTPUT_DIR  = os.path.join(BASE_DIR, "datasets")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def unify_occupancy_scale(df, col="avg_occupancy"):
    if col in df.columns:
        mx = df[col].max(skipna=True)
        if pd.notna(mx) and mx > 1.5:
            df[col] = df[col] / 100.0
    return df

def build_windows(arr_X, arr_y, window=12, horizon=1):
    X_list, y_list = [], []
    for i in range(window, len(arr_X) - horizon + 1):
        X_list.append(arr_X[i - window : i])
        y_list.append(arr_y[i + horizon - 1])
    return np.array(X_list), np.array(y_list)

def time_split_by_ratio(timestamps, train_ratio=0.7, val_ratio=0.15):
    ts_sorted = np.sort(np.unique(timestamps))
    n = len(ts_sorted)
    train_end = ts_sorted[int(n * train_ratio) - 1]
    val_end   = ts_sorted[int(n * (train_ratio + val_ratio)) - 1]

    train_mask = timestamps <= train_end
    val_mask   = (timestamps > train_end) & (timestamps <= val_end)
    test_mask  = timestamps > val_end
    return train_mask, val_mask, test_mask

def main():
    # 1) 讀取 parquet
    files = sorted(glob.glob(os.path.join(CLEANED_DIR, "*.parquet")))
    if not files:
        raise FileNotFoundError(f"在 {CLEANED_DIR} 找不到 parquet 檔")

    dfs = []
    for fp in files:
        df = pd.read_parquet(fp)
        cols = ["time_bin", "vd_id"] + FEATURES
        df = df[[c for c in cols if c in df.columns]].copy()
        dfs.append(df)
    df_all = pd.concat(dfs, ignore_index=True)

    # 2) 篩選 VD
    df_all = df_all[df_all["vd_id"] == VD_ID].copy()

    # 3) 時間排序
    df_all["time_bin"] = pd.to_datetime(df_all["time_bin"], errors="coerce")
    df_all = df_all.dropna(subset=["time_bin"]).sort_values("time_bin").reset_index(drop=True)

    # === 新增時間特徵 ===
    df_all["hour"] = df_all["time_bin"].dt.hour
    df_all["weekday"] = df_all["time_bin"].dt.weekday
    df_all["is_weekend"] = (df_all["weekday"] >= 5).astype(int)
    df_all["is_peak"] = df_all["hour"].apply(lambda h: 1 if (7 <= h <= 9) or (16 <= h <= 19) else 0)

    # 加入週期特徵（sin/cos 週期化時間）
    df_all["hour_sin"] = np.sin(2 * np.pi * df_all["hour"] / 24)
    df_all["hour_cos"] = np.cos(2 * np.pi * df_all["hour"] / 24)
    df_all["weekday_sin"] = np.sin(2 * np.pi * df_all["weekday"] / 7)
    df_all["weekday_cos"] = np.cos(2 * np.pi * df_all["weekday"] / 7)

    # 加入變化率特徵
    df_all["speed_diff"] = df_all["avg_speed"].diff().fillna(0)
    df_all["vehicle_diff"] = df_all["total_vehicles"].diff().fillna(0)

    # 4) 處理佔有率 + 缺值
    df_all = unify_occupancy_scale(df_all, "avg_occupancy")
    df_all = df_all.dropna(subset=FEATURES).reset_index(drop=True)

    # === (新增強) 長期用日粒度輸出 ===
    df_daily = (
        df_all.set_index("time_bin")
            .resample("1D")
            .agg({
                "avg_speed": "mean",
                "total_vehicles": "mean",
                "is_weekend": "max",
                "is_peak": "max"
            })
            .dropna()
            .reset_index()
    )

    # === NEW for Prophet long-term: 移動平均速度 ===
    df_daily["speed_ma7"] = df_daily["avg_speed"].rolling(7, min_periods=1).mean()

    # Prophet 格式需求
    df_daily.rename(columns={"time_bin": "ds", "avg_speed": "y"}, inplace=True)

    daily_csv_path = os.path.join(OUTPUT_DIR, "daily_series.csv")
    df_daily.to_csv(daily_csv_path, index=False, encoding="utf-8")
    print("📄 已輸出 Prophet 用日資料:", daily_csv_path)
    print("    欄位:", list(df_daily.columns))

    # 5) sliding window for LSTM (原本保持)
    feat_df = df_all[FEATURES].astype(float)
    tgt_df  = df_all[TARGETS].astype(float)
    ts = df_all["time_bin"].values

    X_all, y_all = build_windows(feat_df.values, tgt_df.values, window=WINDOW, horizon=HORIZON)
    ts_for_samples = ts[WINDOW - 1 : len(ts) - HORIZON + 0]

    train_m, val_m, test_m = time_split_by_ratio(ts_for_samples, TRAIN_RATIO, VAL_RATIO)

    X_train_raw, y_train_raw = X_all[train_m], y_all[train_m]
    x_mean = X_train_raw.mean(axis=(0,1), keepdims=True)
    x_std  = X_train_raw.std(axis=(0,1), keepdims=True) + 1e-8
    y_mean = y_train_raw.mean(axis=0, keepdims=True)
    y_std  = y_train_raw.std(axis=0, keepdims=True) + 1e-8

    def norm_X(X): return (X - x_mean) / x_std
    def norm_y(y): return (y - y_mean) / y_std

    X_train, y_train = norm_X(X_all[train_m]), norm_y(y_all[train_m])
    X_val, y_val     = norm_X(X_all[val_m]),   norm_y(y_all[val_m])
    X_test, y_test   = norm_X(X_all[test_m]),  norm_y(y_all[test_m])

    vd_safe = VD_ID.replace("/", "_").replace("\\", "_")
    out_path = os.path.join(OUTPUT_DIR, f"traffic_lstm_{vd_safe}.pkl")
    out = {
        "X_train": X_train, "y_train": y_train,
        "X_val":   X_val,   "y_val":   y_val,
        "X_test":  X_test,  "y_test":  y_test,
        "feature_names": FEATURES,
        "target_names":  TARGETS,
        "vd_id": VD_ID,
        "window": WINDOW,
        "horizon": HORIZON,
        "freq": "15min",
        "x_mean": x_mean.squeeze(), "x_std": x_std.squeeze(),
        "y_mean": y_mean.squeeze(), "y_std": y_std.squeeze(),
        "train_time_bin": ts_for_samples[train_m],
        "val_time_bin":   ts_for_samples[val_m],
        "test_time_bin":  ts_for_samples[test_m],
    }
    with open(out_path, "wb") as f:
        pickle.dump(out, f)

    print("✅ 已完成，存到:", out_path)

if __name__ == "__main__":
    main()
