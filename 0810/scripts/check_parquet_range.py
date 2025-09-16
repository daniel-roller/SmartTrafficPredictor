import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ===== 使用者設定 =====
START_DATE = "20240101"   # 起始日
END_DATE   = "20240107"   # 結束日 (含當天)
CLEANED_DIR = os.path.join("..", "cleaned")

# 合理範圍設定
SPEED_MAX = 160.0
OCC_MODE = "auto"   # "auto" | "0-1" | "0-100"
VEH_MAX = 5000


def daterange(start_date: str, end_date: str):
    start = datetime.strptime(start_date, "%Y%m%d")
    end = datetime.strptime(end_date, "%Y%m%d")
    curr = start
    while curr <= end:
        yield curr.strftime("%Y%m%d")
        curr += timedelta(days=1)


def check_day(date_str: str):
    path = os.path.join(CLEANED_DIR, f"{date_str}.parquet")
    if not os.path.exists(path):
        print(f"⚠️ {date_str} parquet 不存在，跳過")
        return

    print(f"\n=== 檢查 {date_str} ===")
    df = pd.read_parquet(path)

    # 去掉時區
    if "time_bin" in df.columns and pd.api.types.is_datetime64_any_dtype(df["time_bin"]):
        try:
            df["time_bin"] = df["time_bin"].dt.tz_convert(None)
        except Exception:
            df["time_bin"] = pd.to_datetime(df["time_bin"], errors="coerce")

    # 缺失值比例
    na_ratio = df.isna().mean()
    print("缺失率：")
    print(na_ratio)

    # 值域檢查
    bad_speed = df[(df["avg_speed"] < 0) | (df["avg_speed"] > SPEED_MAX)]
    occ_max = df["avg_occupancy"].max(skipna=True)
    if OCC_MODE == "auto":
        occ_mode = "0-1" if pd.notna(occ_max) and occ_max <= 1.5 else "0-100"
    else:
        occ_mode = OCC_MODE
    if occ_mode == "0-1":
        bad_occ = df[(df["avg_occupancy"] < 0) | (df["avg_occupancy"] > 1)]
    else:
        bad_occ = df[(df["avg_occupancy"] < 0) | (df["avg_occupancy"] > 100)]
    bad_veh = df[(df["total_vehicles"] < 0) | (df["total_vehicles"] > VEH_MAX)]

    print(f"速度異常筆數: {len(bad_speed)}")
    print(f"佔有率異常筆數: {len(bad_occ)} (尺度={occ_mode})")
    print(f"車流量異常筆數: {len(bad_veh)}")

    print(f"✅ {date_str} 檢查完成，共 {len(df)} 筆")


if __name__ == "__main__":
    for d in daterange(START_DATE, END_DATE):
        check_day(d)
