"""
make_dataset.py
----------------
自動完成：
1. 讀取 select/ 下的所有 CSV
2. 清理資料（僅保留 Speed / Flow，去除缺值與異常）
3. 轉為滑動視窗資料集 (X, y)
4. 輸出可用於訓練深度學習模型的 numpy 檔案
"""

import os
import glob
import numpy as np
import pandas as pd

# === 參數設定 ===
INPUT_DIR = "select"
CLEAN_DIR = "cleaned"
OUTPUT_DIR = "datasets"

WINDOW = 12    # 用前12筆資料預測下一筆（1小時）
HORIZON = 1    # 預測1步（5分鐘）
TARGET = "speed"   # 預測速度 (speed)

# === 資料夾建立 ===
os.makedirs(CLEAN_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def clean_dataframe(df):
    """
    只保留 Speed / Flow 兩欄並清理缺值與異常值
    """
    # 嘗試找出 speed / flow 欄位
    possible_speed = [c for c in df.columns if "speed" in c.lower() or "速" in c]
    possible_flow = [c for c in df.columns if "flow" in c.lower() or "流" in c]

    if not possible_speed or not possible_flow:
        raise ValueError("❌ 無法找到 Speed 或 Flow 欄位")

    speed_col = possible_speed[0]
    flow_col = possible_flow[0]

    df = df[[speed_col, flow_col]].copy()
    df.columns = ["speed", "flow"]

    # 去除缺值與異常（speed<=0 或 flow<0）
    df.replace([-1, 9999], np.nan, inplace=True)
    df = df[(df["speed"] > 0) & (df["flow"] >= 0)]
    df = df.dropna().reset_index(drop=True)

    return df


def create_sequences(data, window, horizon):
    """
    將時間序列資料轉為訓練集 (X, y)
    data: numpy array [speed, flow]
    X shape: (樣本數, window, 特徵數)
    y shape: (樣本數,)
    """
    X, y = [], []
    for i in range(len(data) - window - horizon):
        X.append(data[i:i+window])
        y.append(data[i+window+horizon-1, 0])  # 預測speed
    return np.array(X), np.array(y)


# === 主流程 ===
for f in glob.glob(os.path.join(INPUT_DIR, "*.csv")):
    try:
        print(f"\n📂 處理檔案：{f}")
        df = pd.read_csv(f)
        df = clean_dataframe(df)

        # 儲存清理後資料
        clean_path = os.path.join(CLEAN_DIR, os.path.basename(f))
        df.to_csv(clean_path, index=False)
        print(f"✅ 已清理並儲存至：{clean_path}，筆數={len(df)}")

        # 製作滑動視窗資料
        data = df[["speed", "flow"]].values
        X, y = create_sequences(data, WINDOW, HORIZON)
        print(f"🧩 建立資料集：X={X.shape}, y={y.shape}")

        # 儲存 numpy 檔
        base = os.path.splitext(os.path.basename(f))[0]
        np.save(os.path.join(OUTPUT_DIR, f"{base}_X.npy"), X)
        np.save(os.path.join(OUTPUT_DIR, f"{base}_y.npy"), y)
        print(f"💾 輸出至 datasets/{base}_X.npy, {base}_y.npy")

    except Exception as e:
        print(f"⚠️ {os.path.basename(f)} 處理失敗：{e}")

print("\n🎯 所有資料已完成清理與滑動視窗化！")
