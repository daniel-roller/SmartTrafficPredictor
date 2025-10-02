import os
import gzip
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
import lxml.etree as ET
from tqdm import tqdm
from datetime import datetime, timedelta


# === 使用者可調參數 ===
START_DATE = "20240901"   # 起始日 (YYYYMMDD)
END_DATE   = "20241031"   # 結束日 (YYYYMMDD)，包含這一天
MAX_WORKERS = 16          # 平行處理核心數


# === 資料夾結構 ===
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_BASE = os.path.join(BASE_DIR, "raw")
CLEANED_BASE = os.path.join(BASE_DIR, "cleaned")
os.makedirs(CLEANED_BASE, exist_ok=True)


def strip_ns(tree):
    """移除 XML namespace"""
    for elem in tree.getiterator():
        if not hasattr(elem.tag, "find"):
            continue
        i = elem.tag.find("}")
        if i >= 0:
            elem.tag = elem.tag[i + 1 :]
    return tree


def parse_xml_file(gz_path):
    """解壓縮並解析單一 .gz 檔案"""
    rows = []
    try:
        with gzip.open(gz_path, "rb") as f:
            content = f.read()

        root = ET.fromstring(content)
        root = strip_ns(root)

        vdlives = root.findall(".//VDLive")
        if not vdlives:
            return rows

        for vd in vdlives:
            vd_id = vd.findtext("VDID")
            status = vd.findtext("Status")
            time = vd.findtext("DataCollectTime")

            for linkflow in vd.findall(".//LinkFlow"):
                for lane in linkflow.findall(".//Lane"):
                    lane_id = lane.findtext("LaneID")
                    lane_type = lane.findtext("LaneType")
                    speed = lane.findtext("Speed")
                    occupancy = lane.findtext("Occupancy")
                    vehicle_count = len(lane.findall(".//Vehicle"))

                    rows.append([
                        time, vd_id, status,
                        lane_id, lane_type,
                        speed, occupancy, vehicle_count
                    ])
    except Exception:
        return rows
    return rows


def process_day(date_str: str, max_workers=16):
    """處理單日所有 .gz 檔案，平行解析並輸出 Parquet"""
    day_dir = os.path.join(RAW_BASE, date_str)
    if not os.path.exists(day_dir):
        print(f"⚠️ 找不到 {day_dir}，跳過 {date_str}")
        return

    files = sorted([os.path.join(day_dir, f) for f in os.listdir(day_dir) if f.endswith(".gz")])
    all_rows, skipped = [], []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(parse_xml_file, f): f for f in files}
        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"解壓縮 {date_str}"):
            result = fut.result()
            if not result:
                skipped.append(futures[fut])
            all_rows.extend(result)

    if skipped:
        skipped_path = os.path.join(CLEANED_BASE, f"{date_str}_skipped.txt")
        with open(skipped_path, "w", encoding="utf-8") as f:
            for s in skipped:
                f.write(s + "\n")
        print(f"⚠️ {len(skipped)} 個檔案沒有資料，已寫入 {skipped_path}")

    if all_rows:
        df_day = pd.DataFrame(
            all_rows,
            columns=["time", "vd_id", "status", "lane_id", "lane_type", "speed", "occupancy", "vehicle_count"]
        )

        # 基本清洗
        df_day.replace({"-99": None, "-1": None}, inplace=True)
        df_day["speed"] = pd.to_numeric(df_day["speed"], errors="coerce")
        df_day["occupancy"] = pd.to_numeric(df_day["occupancy"], errors="coerce")
        df_day["vehicle_count"] = pd.to_numeric(df_day["vehicle_count"], errors="coerce")
        df_day["time"] = pd.to_datetime(df_day["time"], errors="coerce")

        # 聚合：每 VD，每 15 分鐘
        df_day = df_day.dropna(subset=["time", "vd_id"])
        df_day["time_bin"] = df_day["time"].dt.floor("15min")
        agg_df = df_day.groupby(["time_bin", "vd_id"]).agg(
            avg_speed=("speed", "mean"),
            avg_occupancy=("occupancy", "mean"),
            total_vehicles=("vehicle_count", "sum")
        ).reset_index()

        out_path = os.path.join(CLEANED_BASE, f"{date_str}.parquet")
        agg_df.to_parquet(out_path, index=False)
        print(f"✅ 已輸出 {out_path}，共 {len(agg_df)} 筆聚合紀錄")
    else:
        print(f"⚠️ {date_str} 沒有成功產生資料")


def daterange(start_date: str, end_date: str):
    start = datetime.strptime(start_date, "%Y%m%d")
    end = datetime.strptime(end_date, "%Y%m%d")
    curr = start
    while curr <= end:
        yield curr.strftime("%Y%m%d")
        curr += timedelta(days=1)


if __name__ == "__main__":
    for d in daterange(START_DATE, END_DATE):
        process_day(d, max_workers=MAX_WORKERS)
