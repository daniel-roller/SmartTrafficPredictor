import pandas as pd
import numpy as np
import glob, os, shutil

# === 1. 目標路段（起點, 終點） ===
targets = [
    ("圓山", "三重"),
    ("后里", "台中"),
    ("路竹", "岡山"),
    ("新店", "土城"),
    ("堤頂", "五股"),
    ("楊梅", "新竹"),
    ("路竹", "高科"),
    ("鶯歌", "高原"),
    ("烏日", "草屯"),
    ("南港", "頭城"),
]

# === 2. 建立資料夾 ===
os.makedirs("select", exist_ok=True)
os.makedirs("results", exist_ok=True)

# === 3. 掃描 raw/ 所有 CSV，篩出符合起點+終點的檔案 ===
files = sorted(glob.glob("raw/*.csv"))
target_files = []
for f in files:
    for start, end in targets:
        if start in f and end in f:
            target_files.append(f)
            break

print(f"🔍 找到 {len(target_files)} 個符合指定路段的 CSV 檔案")

if not target_files:
    print("⚠️ 沒有找到符合條件的檔案，請確認檔名或字詞是否一致")
    exit()

# === 4. 檢查每個檔案的資料品質 ===
results = []

def detect_time_col(cols):
    for c in cols:
        if any(x in c for x in ["時", "Time", "time"]):
            return c
    return None

def detect_speed_col(cols):
    for c in cols:
        if any(x in c for x in ["速", "Speed", "speed"]):
            return c
    return None

for f in target_files:
    try:
        df = pd.read_csv(f)
        cols = list(df.columns)
        time_col = detect_time_col(cols)
        speed_col = detect_speed_col(cols)
        if time_col is None or speed_col is None:
            print(f"⚠️ {os.path.basename(f)} 缺少時間或速度欄位，略過")
            continue

        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
        df = df.sort_values(time_col).dropna(subset=[time_col])

        total = len(df)
        time_diff = df[time_col].diff().dt.total_seconds().dropna() / 60
        long_gaps = (time_diff > 5).sum()
        missing_ratio = df.isna().mean().mean()
        std_speed = df[speed_col].replace(-1, np.nan).std(skipna=True)
        date_range = (df[time_col].max() - df[time_col].min()).days

        results.append({
            "file": os.path.basename(f),
            "rows": total,
            "missing_ratio": round(missing_ratio, 4),
            "time_gaps": int(long_gaps),
            "speed_std": round(std_speed, 2),
            "date_range_days": date_range
        })
    except Exception as e:
        print(f"⚠️ {os.path.basename(f)} 讀取失敗：{e}")

# === 5. 產出結果報表 ===
summary = pd.DataFrame(results)
if summary.empty:
    print("⚠️ 沒有成功分析任何檔案。")
    exit()

summary = summary.sort_values(
    ["missing_ratio", "time_gaps", "speed_std"],
    ascending=[True, True, False]
)
summary.to_csv("results/road_quality_summary.csv", index=False)

print("✅ 已輸出 results/road_quality_summary.csv")

# === 6. 挑選前 5 名最乾淨的路段 ===
top5 = summary.head(5)
print("\n📈 最推薦的五個路段：")
print(top5[["file", "missing_ratio", "time_gaps", "speed_std", "date_range_days"]])

# === 7. 複製到 select/ ===
for fname in top5["file"]:
    src = os.path.join("raw", fname)
    dst = os.path.join("select", fname)
    if os.path.exists(src):
        shutil.copy(src, dst)

print("\n✅ 已將前五個路段 CSV 檔案複製到 select/ 資料夾")
