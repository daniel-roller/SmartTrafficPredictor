import os
import sys
import time
import requests
from tqdm import tqdm
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "*/*",
    "Connection": "keep-alive",
}


def download_file(url, save_path, retries=3):
    if os.path.exists(save_path):
        return "skip"

    for attempt in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=30)
            if r.status_code == 200:
                with open(save_path, "wb") as f:
                    f.write(r.content)
                return "success"
            else:
                return f"fail_{r.status_code}"
        except requests.exceptions.Timeout:
            time.sleep(1)
            return "fail_timeout"
        except Exception as e:
            time.sleep(1)
            return f"fail_error:{str(e)}"

    return "fail_unknown"


def generate_filenames():
    """產生正確的一天 1440 個檔名 (HHMM 格式)"""
    names = []
    for h in range(24):
        for m in range(60):
            names.append(f"{h:02d}{m:02d}")
    return names


def download_vd_day(date_str, max_workers=8, max_rounds=3):
    """下載單日 VD 資料"""
    base_dir = os.path.join(os.path.dirname(__file__), "..", "raw")
    base_dir = os.path.abspath(base_dir)
    save_dir = os.path.join(base_dir, date_str)
    os.makedirs(save_dir, exist_ok=True)

    base_url = f"https://tisvcloud.freeway.gov.tw/history/motc20/VD/{date_str}/"
    filenames = generate_filenames()
    urls = [(f"{base_url}VDLive_{name}.xml.gz", os.path.join(save_dir, f"VDLive_{name}.xml.gz"))
            for name in filenames]

    all_failed = []

    for round_id in range(1, max_rounds + 1):
        success, fail, skip = 0, 0, 0
        fail_list = []
        fail_reasons = {}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(download_file, url, save_path): (url, save_path) for url, save_path in urls}
            for f in tqdm(as_completed(futures), total=len(futures), desc=f"{date_str} Round {round_id}"):
                result = f.result()
                if result == "success":
                    success += 1
                elif result == "skip":
                    skip += 1
                else:
                    fail += 1
                    fail_list.append((futures[f][0], futures[f][1], result))
                    fail_reasons[result] = fail_reasons.get(result, 0) + 1

        print(f"\n📊 {date_str} 第 {round_id} 輪結果: 成功 {success}, 跳過 {skip}, 失敗 {fail}")
        if fail_reasons:
            print("   失敗原因統計：")
            for reason, count in fail_reasons.items():
                print(f"   - {reason}: {count}")

        if fail == 0:
            print(f"🎉 {date_str} 全部下載成功！")
            break
        else:
            print(f"⚠️ {date_str} 還有 {fail} 筆失敗，準備補抓...")
            urls = [(url, save_path) for url, save_path, _ in fail_list]
            all_failed = fail_list

    if all_failed:
        failed_txt = os.path.join(save_dir, "failed.txt")
        with open(failed_txt, "w", encoding="utf-8") as f:
            for url, save_path, reason in all_failed:
                f.write(f"{url}\t{reason}\n")
        print(f"❌ {date_str} 仍有 {len(all_failed)} 筆失敗，已存到 {failed_txt}")

    print(f"📂 {date_str} 存放在: {save_dir}")


def download_vd_range(start_date, end_date, max_workers=8, max_rounds=3):
    """下載一段日期範圍內的 VD 資料"""
    start = datetime.strptime(start_date, "%Y%m%d")
    end = datetime.strptime(end_date, "%Y%m%d")
    delta = timedelta(days=1)

    current = start
    while current <= end:
        date_str = current.strftime("%Y%m%d")
        download_vd_day(date_str, max_workers=max_workers, max_rounds=max_rounds)
        current += delta


if __name__ == "__main__":
    if len(sys.argv) == 2:
        # 單日下載
        download_vd_day(sys.argv[1], max_workers=8, max_rounds=3)
    elif len(sys.argv) == 3:
        # 多日下載
        download_vd_range(sys.argv[1], sys.argv[2], max_workers=8, max_rounds=3)
    else:
        print("用法：")
        print("  python download_vd_data_range.py 20240101         # 單日下載")
        print("  python download_vd_data_range.py 20240101 20240107 # 多日下載")

# python download_vd_data_range.py 20240101 20240107