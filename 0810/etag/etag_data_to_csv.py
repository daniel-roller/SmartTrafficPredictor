import os
import re
import time
import pandas as pd
import xml.etree.ElementTree as ET
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from collections import defaultdict

# ===== 設定 =====
data_folder = "etag_data"      # 存放一週資料的資料夾
output_csv = "eTag_加權統計_一週.csv"
missing_csv = "missing_files.csv"
error_log = "parse_errors.log"

VEHICLE_WEIGHT = {
    "31": 1.0,  # 小型車
    "41": 1.5,  # 大型車
    "42": 1.5,
    "51": 1.5,  # 聯結車
    "52": 1.5
}

# 預期時間點（全天）
EXPECTED_TIMES = [f"{h:02d}{m:02d}" for h in range(24) for m in range(0, 60, 5)]

def save_csv_with_progress(df, output_csv, chunksize=100000):
    total_rows = len(df)
    print(f"✅ 解析完成，開始分批寫入 CSV（共 {total_rows} 筆資料）...")

    t_write_start = time.time()
    with open(output_csv, "w", encoding="utf-8-sig", newline="") as f:
        for i, chunk_start in enumerate(range(0, total_rows, chunksize), start=1):
            chunk_end = min(chunk_start + chunksize, total_rows)
            df.iloc[chunk_start:chunk_end].to_csv(f, index=False, header=(i == 1))
            print(f"📄 已寫入 {chunk_end}/{total_rows} 筆 ({(chunk_end/total_rows)*100:.2f}%)")

    print(f"✅ CSV 寫入完成，總耗時 {time.time() - t_write_start:.2f} 秒")


# ===== 解析單檔 XML =====
def parse_xml(args):
    day_folder, filepath = args
    records = []
    errors = None
    try:
        tree = ET.parse(filepath)
        root = tree.getroot()
        # 檔名時間
        file_name = os.path.basename(filepath)
        time_match = re.search(r"(\d{4})", file_name)
        time_str = time_match.group(1) if time_match else "0000"
        date_str = day_folder.split("_")[1]
        time_fmt = f"{date_str} {time_str[:2]}:{time_str[2:]}:00"

        for live in root.findall(".//{*}ETagPairLive"):
            pair_id = live.findtext(".//{*}ETagPairId", "")
            start, end = pair_id.split("-") if "-" in pair_id else ("", "")
            for flow in live.findall(".//{*}Flow"):
                vtype = flow.findtext(".//{*}VehicleType", "")
                ttime = float(flow.findtext(".//{*}TravelTime", "0"))
                speed = float(flow.findtext(".//{*}SpaceMeanSpeed", "0"))
                count = int(flow.findtext(".//{*}VehicleCount", "0"))
                weight = VEHICLE_WEIGHT.get(vtype, 0)
                records.append({
                    "DataCollectTime": time_fmt,
                    "StartLocation": start,
                    "EndLocation": end,
                    "VehicleType": vtype,
                    "Volume": count * weight,
                    "Speed": speed,
                    "TripTime": ttime
                })
    except Exception as e:
        errors = f"{filepath}: {e}"
    return records, errors

# ===== 主程式 =====
if __name__ == "__main__":
    t_start = time.time()
    xml_files = []
    missing_files = defaultdict(list)

    # 收集 XML 路徑 + 檢查缺檔
    for day_folder in sorted(os.listdir(data_folder)):
        folder_path = os.path.join(data_folder, day_folder)
        if not os.path.isdir(folder_path):
            continue
        files = sorted(f for f in os.listdir(folder_path) if f.endswith(".xml"))
        file_times = {re.search(r"(\d{4})", f).group(1) for f in files}
        # 找缺檔
        for t in EXPECTED_TIMES:
            if t not in file_times:
                missing_files[day_folder].append(t)
        # 收集完整路徑
        for file in files:
            xml_files.append((day_folder, os.path.join(folder_path, file)))

    print(f"📂 發現 {len(xml_files)} 個 XML 檔案，開始解析（多核）...\n")

    records_all = []
    errors_all = []

    # 多核解析
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(parse_xml, args): args for args in xml_files}
        for future in tqdm(as_completed(futures), total=len(futures), desc="📖 解析中", unit="file"):
            records, errors = future.result()
            if records:
                records_all.extend(records)
            if errors:
                errors_all.append(errors)

    # 寫錯誤日誌
    if errors_all:
        with open(error_log, "w", encoding="utf-8") as f:
            for line in errors_all:
                f.write(line + "\n")
        print(f"⚠️ 有 {len(errors_all)} 筆解析錯誤，已寫入 {error_log}")

    # 寫缺檔清單
    if missing_files:
        with open(missing_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["DayFolder", "MissingTime"])
            for day, times in missing_files.items():
                for t in times:
                    writer.writerow([day, t])
        print(f"⚠️ 有缺檔，已輸出 {missing_csv}")

    # 轉成 DataFrame
    df = pd.DataFrame(records_all)

    # 安全加權平均
    def safe_weighted_avg(x, col):
        weights = df.loc[x.index, "Volume"]
        total = weights.sum()
        return (x * weights).sum() / total if total > 0 else 0

    # 彙總 + Coverage
    summary = df.groupby(["DataCollectTime", "StartLocation", "EndLocation"]).agg({
        "Volume": "sum",
        "Speed": lambda x: safe_weighted_avg(x, "Speed"),
        "TripTime": lambda x: safe_weighted_avg(x, "TripTime"),
        "VehicleType": "count"  # 用來計算 coverage
    }).reset_index()

    summary.rename(columns={
        "Volume": "WeightedVolume",
        "Speed": "WeightedAvgSpeed",
        "TripTime": "WeightedAvgTripTime",
        "VehicleType": "RecordsCount"
    }, inplace=True)

    # 加 coverage 欄位
    summary["Coverage"] = summary["RecordsCount"] / 3
    summary.drop(columns=["RecordsCount"], inplace=True)

    # 改用分批寫入方式
    save_csv_with_progress(summary, output_csv)
    print(f"🎯 全部流程完成，總耗時 {time.time() - t_start:.2f} 秒")

