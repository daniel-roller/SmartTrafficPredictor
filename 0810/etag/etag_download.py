import os
import requests
import gzip
import shutil
from datetime import datetime, timedelta
from time import sleep
import zipfile
from tqdm import tqdm
import csv

# ======= 設定 =======
start_date = "20250616"  # 開始日期 (YYYYMMDD)
end_date   = "20250622"  # 結束日期 (YYYYMMDD)
save_base  = "etag_data" # 儲存資料夾
retry_times = 3          # 單檔下載重試次數
timeout_sec = 5          # 單檔 timeout 秒數
hours_range = range(0, 24)  # 抓取小時範圍（例：range(6, 22) 只抓 06:00-21:59）
fail_log = "failed_files.csv"  # 下載失敗記錄檔

# ======= 建立日期列表 =======
def get_date_list(start, end):
    s = datetime.strptime(start, "%Y%m%d")
    e = datetime.strptime(end, "%Y%m%d")
    dates = []
    while s <= e:
        dates.append(s.strftime("%Y%m%d"))
        s += timedelta(days=1)
    return dates

dates = get_date_list(start_date, end_date)

# ======= 單檔下載並解壓 =======
def download_and_extract(date, time_str, folder):
    gz_name = f"ETagPairLive_{time_str}.xml.gz"
    xml_name = gz_name[:-3]
    gz_path = os.path.join(folder, gz_name)
    xml_path = os.path.join(folder, xml_name)
    url = f"https://tisvcloud.freeway.gov.tw/history/motc20/ETag/{date}/{gz_name}"

    # 檔案已存在就跳過
    if os.path.exists(xml_path) and os.path.getsize(xml_path) > 0:
        return True, None

    wait_time = 0.5
    for attempt in range(1, retry_times+1):
        try:
            r = requests.get(url, timeout=timeout_sec)
            if r.status_code == 200 and len(r.content) > 0:
                with open(gz_path, "wb") as f:
                    f.write(r.content)
                # 解壓
                with gzip.open(gz_path, "rb") as f_in, open(xml_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
                os.remove(gz_path)
                return True, None
            else:
                sleep(wait_time)
                wait_time *= 2  # 指數回退
        except requests.exceptions.RequestException as e:
            sleep(wait_time)
            wait_time *= 2
    return False, url  # 回傳失敗的 URL

# ======= 載入舊的失敗清單（補抓用） =======
def load_failed_list():
    if os.path.exists(fail_log):
        with open(fail_log, newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            return [row[0] for row in reader if row]
    return []

# ======= 主程式 =======
os.makedirs(save_base, exist_ok=True)
failed_urls = []
old_failed = load_failed_list()

# 如果有舊的失敗清單，優先補抓
if old_failed:
    print(f"📄 發現舊的失敗清單，共 {len(old_failed)} 筆，開始補抓...")
    with tqdm(total=len(old_failed), desc="⬇ 補抓中", unit="file") as pbar:
        for url in old_failed:
            parts = url.split("/")
            date = parts[-2]
            gz_name = parts[-1]
            time_str = gz_name.split("_")[1].split(".")[0]
            folder = os.path.join(save_base, f"etag_{date}")
            os.makedirs(folder, exist_ok=True)
            ok, fail_url = download_and_extract(date, time_str, folder)
            if not ok:
                failed_urls.append(fail_url)
            pbar.update(1)

# 正常批次下載
total_files = len(dates) * len(hours_range) * 12  # 每小時 12 檔（5 分鐘一檔）
with tqdm(total=total_files, desc="⬇ 下載中", unit="file") as pbar:
    for date in dates:
        folder = os.path.join(save_base, f"etag_{date}")
        os.makedirs(folder, exist_ok=True)
        for hour in hours_range:
            for minute in range(0, 60, 5):
                time_str = f"{hour:02d}{minute:02d}"
                ok, fail_url = download_and_extract(date, time_str, folder)
                if not ok:
                    failed_urls.append(fail_url)
                pbar.update(1)

# 輸出失敗清單
if failed_urls:
    with open(fail_log, "w", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for url in failed_urls:
            writer.writerow([url])
    print(f"⚠️ 有 {len(failed_urls)} 筆下載失敗，已記錄到 {fail_log}")
else:
    if os.path.exists(fail_log):
        os.remove(fail_log)
    print("✅ 全部檔案下載成功，無失敗項目")

# ======= coverage 檢查 =======
expected_per_day = len(hours_range) * 12
for date in dates:
    folder = os.path.join(save_base, f"etag_{date}")
    xmls = [f for f in os.listdir(folder) if f.endswith(".xml")]
    coverage = len(xmls) / expected_per_day
    print(f"{date}: 檔案數={len(xmls)} / {expected_per_day}，Coverage={coverage:.2%}")

# ======= 壓縮成 ZIP =======
zip_name = "etag_week.zip"
with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as zipf:
    for root, dirs, files in os.walk(save_base):
        for file in files:
            file_path = os.path.join(root, file)
            arc_name = os.path.relpath(file_path, save_base)
            zipf.write(file_path, arc_name)
print(f"📦 已壓縮成 {zip_name}")
