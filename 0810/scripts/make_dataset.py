# make_dataset.py
import os
import glob
import pickle
import numpy as np
import pandas as pd

# === 自動設定資料夾路徑 ===
BASE_DIR    = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CLEANED_DIR = os.path.join(BASE_DIR, "cleaned")
OUTPUT_DIR  = os.path.join(BASE_DIR, "datasets")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === 固定使用指定路段的 VD ===
VD_ID       = "VD-N1-N-0.000-M-LOOP"  # 國1 圓山端
WINDOW      = 12
HORIZON     = 1
FEATURES    = ["avg_speed", "avg_occupancy", "total_vehicles"]
TARGETS     = ["avg_speed", "total_vehicles"]
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15

def unify_occupancy_scale(df, col="avg_occupancy"):
    """把佔有率統一到 0~1"""
    if col in df.columns:
        mx = df[col].max(skipna=True)
        if pd.notna(mx) and mx > 1.5:
            df[col] = df[col] / 100.0
    return df

def build_windows(arr_X, arr_y, window=12, horizon=1):
    """轉換成 (X, y) 樣本"""
    X_list, y_list = [], []
    for i in range(window, len(arr_X) - horizon + 1):
        X_list.append(arr_X[i - window : i])
        y_list.append(arr_y[i + horizon - 1])
    return np.array(X_list), np.array(y_list)

def time_split_by_ratio(timestamps, train_ratio=0.7, val_ratio=0.15):
    """依時間分割 train/val/test"""
    ts_sorted = np.sort(np.unique(timestamps))
    n = len(ts_sorted)
    train_end = ts_sorted[int(n * train_ratio) - 1]
    val_end   = ts_sorted[int(n * (train_ratio + val_ratio)) - 1]

    train_mask = timestamps <= train_end
    val_mask   = (timestamps > train_end) & (timestamps <= val_end)
    test_mask  = timestamps > val_end
    return train_mask, val_mask, test_mask

def main():
    # 1) 讀取所有 parquet
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

    # 2) 篩選指定 VD
    if VD_ID not in df_all["vd_id"].unique():
        raise ValueError(f"⚠️ 找不到 VD_ID={VD_ID}，請確認資料中是否存在")

    df_all = df_all[df_all["vd_id"] == VD_ID].copy()

    # 3) 時間排序
    df_all["time_bin"] = pd.to_datetime(df_all["time_bin"], errors="coerce")
    df_all = df_all.dropna(subset=["time_bin"]).sort_values("time_bin").reset_index(drop=True)

    # 4) 處理佔有率 + 缺值
    df_all = unify_occupancy_scale(df_all, "avg_occupancy")
    df_all = df_all.dropna(subset=FEATURES).reset_index(drop=True)

    # 5) 特徵與目標
    feat_df = df_all[FEATURES].astype(float)
    tgt_df  = df_all[TARGETS].astype(float)
    ts = df_all["time_bin"].values

    # 6) sliding window
    X_all, y_all = build_windows(feat_df.values, tgt_df.values,
                                 window=WINDOW, horizon=HORIZON)
    ts_for_samples = ts[WINDOW - 1 : len(ts) - HORIZON + 0]

    # 7) 時間切割
    train_m, val_m, test_m = time_split_by_ratio(ts_for_samples, TRAIN_RATIO, VAL_RATIO)

    # 8) 標準化
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

    # 9) 存成 pkl（檔名包含 VD）
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
    }
    with open(out_path, "wb") as f:
        pickle.dump(out, f)

    print("✅ 已完成，存到:", out_path)
    print("Shapes:",
          "\n  X_train:", X_train.shape, "y_train:", y_train.shape,
          "\n  X_val:",   X_val.shape,   "y_val:",   y_val.shape,
          "\n  X_test:",  X_test.shape,  "y_test:",  y_test.shape)

if __name__ == "__main__":
    main()
