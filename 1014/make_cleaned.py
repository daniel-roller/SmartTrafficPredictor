import os
import glob
import pandas as pd

# === 修正路徑設定 ===
# 取得當前檔案所在目錄 (1014/)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "raw")
OUTPUT_DIR = os.path.join(BASE_DIR, "cleaned")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def safe_filename(name: str):
    """移除不合法字元"""
    return name.replace("/", "_").replace("\\", "_").replace(" ", "_")

def extract_vd_id(filename: str):
    """從檔名中取出路段名稱"""
    base = os.path.basename(filename)
    name = os.path.splitext(base)[0]
    return name

def read_and_clean(filepath: str):
    """讀取單一 CSV 並清理成標準格式"""
    print(f"📂 處理中：{os.path.basename(filepath)}")

    # 嘗試多種編碼讀取CSV
    encodings = ['utf-8', 'big5', 'cp950', 'gbk']
    df = None
    
    for encoding in encodings:
        try:
            df = pd.read_csv(filepath, encoding=encoding)
            print(f"   ✅ 使用編碼: {encoding}")
            break
        except (UnicodeDecodeError, UnicodeError):
            continue
    
    if df is None:
        raise ValueError(f"❌ 無法讀取檔案 {filepath}，嘗試了所有編碼")

    print(f"   📊 原始欄位: {list(df.columns)}")
    print(f"   📊 原始資料筆數: {len(df)}")
    
    # 標準化欄位名稱
    original_columns = df.columns.tolist()
    df.columns = [c.strip().lower() for c in df.columns]

    # 確認主要欄位
    possible_cols = {
        "speed": None,
        "flow": None,
        "datetime": None
    }
    
    for c in df.columns:
        if "speed" in c:
            possible_cols["speed"] = c
        elif "flow" in c:
            possible_cols["flow"] = c
        elif "time" in c or "date" in c:
            possible_cols["datetime"] = c

    print(f"   🔍 找到欄位對應: {possible_cols}")
    
    # 檢查必要欄位
    missing_cols = [k for k, v in possible_cols.items() if v is None]
    if missing_cols:
        print(f"   ⚠️ 缺少欄位: {missing_cols}")
        print(f"   📋 可用欄位: {list(df.columns)}")
        
        # 嘗試手動對應常見欄位名稱
        if possible_cols["datetime"] is None:
            for col in df.columns:
                if 'datatime' in col or 'time' in col:
                    possible_cols["datetime"] = col
                    break
        
        if possible_cols["speed"] is None:
            for col in df.columns:
                if col in ['speed']:
                    possible_cols["speed"] = col
                    break
                    
        if possible_cols["flow"] is None:
            for col in df.columns:
                if col in ['flow']:
                    possible_cols["flow"] = col
                    break
    
    # 再次檢查
    if not all(possible_cols.values()):
        missing = [k for k, v in possible_cols.items() if v is None]
        raise ValueError(f"❌ 仍然缺少欄位: {missing}")

    # 取出必要欄位
    required_cols = [possible_cols["datetime"], possible_cols["speed"], possible_cols["flow"]]
    df = df[required_cols].copy()
    df.columns = ["time_bin", "avg_speed", "total_vehicles"]

    # 轉換時間格式
    print(f"   ⏰ 轉換時間格式...")
    df["time_bin"] = pd.to_datetime(df["time_bin"], errors="coerce")
    
    # 檢查時間轉換結果
    invalid_time = df["time_bin"].isna().sum()
    if invalid_time > 0:
        print(f"   ⚠️ 有 {invalid_time} 筆資料時間格式無效")
    
    df = df.dropna(subset=["time_bin"]).sort_values("time_bin").reset_index(drop=True)

    # 檢查數值欄位
    print(f"   🔢 清理數值資料...")
    print(f"      速度範圍: {df['avg_speed'].min():.1f} ~ {df['avg_speed'].max():.1f}")
    print(f"      車流量範圍: {df['total_vehicles'].min():.1f} ~ {df['total_vehicles'].max():.1f}")
    
    # 移除不合理數值
    before_clean = len(df)
    df = df[(df["avg_speed"] > 0) & (df["avg_speed"] < 200)]
    df = df[(df["total_vehicles"] >= 0)]
    after_clean = len(df)
    
    if before_clean != after_clean:
        print(f"   🧹 移除異常值: {before_clean} -> {after_clean} 筆")

    # 轉為每小時平均
    print(f"   📊 轉換為每小時平均...")
    df = df.set_index("time_bin").resample("1H").mean().reset_index()

    # 新增 vd_id 欄位
    df["vd_id"] = extract_vd_id(filepath)
    
    print(f"   ✅ 清理完成: {len(df)} 筆每小時資料")
    return df

def main():
    print(f"🔍 搜尋目錄: {RAW_DIR}")
    
    # 檢查raw目錄是否存在
    if not os.path.exists(RAW_DIR):
        print(f"❌ 目錄不存在: {RAW_DIR}")
        return
    
    csv_files = sorted(glob.glob(os.path.join(RAW_DIR, "*.csv")))
    print(f"📁 找到 {len(csv_files)} 個 CSV 檔案")
    
    if not csv_files:
        # 列出raw目錄中的所有檔案幫助除錯
        all_files = os.listdir(RAW_DIR) if os.path.exists(RAW_DIR) else []
        print(f"📋 raw目錄中的所有檔案: {all_files}")
        raise FileNotFoundError(f"❌ 在 {RAW_DIR} 找不到任何 CSV 檔")
    
    print(f"📋 待處理檔案:")
    for i, fp in enumerate(csv_files, 1):
        filename = os.path.basename(fp)
        size_mb = os.path.getsize(fp) / (1024*1024)
        print(f"   {i}. {filename} ({size_mb:.1f} MB)")

    successful = 0
    failed = 0
    
    for fp in csv_files:
        try:
            df_clean = read_and_clean(fp)

            out_name = safe_filename(extract_vd_id(fp)) + ".parquet"
            out_path = os.path.join(OUTPUT_DIR, out_name)

            df_clean.to_parquet(out_path, index=False)
            print(f"✅ 已輸出：{out_name}")
            successful += 1

        except Exception as e:
            print(f"❌ 無法處理 {os.path.basename(fp)}: {e}")
            failed += 1
            import traceback
            traceback.print_exc()

    print(f"\n🎉 清理完成!")
    print(f"   ✅ 成功: {successful} 個檔案")
    print(f"   ❌ 失敗: {failed} 個檔案")
    print(f"   📁 輸出目錄: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()