# debug
import os
import sys
import gzip
import lxml.etree as ET
import pandas as pd


# === 資料夾結構 ===
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_BASE = os.path.join(BASE_DIR, "raw")


def strip_ns(tree):
    """移除 XML namespace"""
    for elem in tree.getiterator():
        if not hasattr(elem.tag, "find"):
            continue
        i = elem.tag.find("}")
        if i >= 0:
            elem.tag = elem.tag[i + 1 :]
    return tree


def parse_xml(xml_content: bytes, fname=""):
    """解析單一 VD XML.gz，回傳 list of rows"""
    rows = []
    try:
        root = ET.fromstring(xml_content)
        root = strip_ns(root)

        vdlives = root.findall(".//VDLive")
        if not vdlives:
            print(f"⚠️ {fname} 沒有 <VDLive> 資料")
            return rows

        for vd in vdlives:
            vd_id = vd.findtext("VDID")
            status = vd.findtext("Status")
            time = vd.findtext("DataCollectTime")

            for linkflow in vd.findall(".//LinkFlow"):
                for laneflow in linkflow.findall(".//LaneFlow"):
                    lane_id = laneflow.findtext("LaneID")
                    speed = laneflow.findtext("Speed")
                    volume = laneflow.findtext("Volume")
                    rows.append([time, vd_id, status, lane_id, speed, volume])

    except Exception as e:
        print(f"❌ 解析失敗 {fname}: {e}")
    return rows


def debug_file(date_str: str, filename: str):
    """單檔除錯用，印出前幾筆 row"""
    gz_path = os.path.join(RAW_BASE, date_str, filename)
    if not os.path.exists(gz_path):
        print(f"❌ 找不到檔案 {gz_path}")
        return

    try:
        with gzip.open(gz_path, "rb") as f:
            content = f.read()
        rows = parse_xml(content, filename)
    except Exception as e:
        print(f"❌ 無法讀取 {gz_path}: {e}")
        return

    if rows:
        df = pd.DataFrame(rows, columns=["time", "vd_id", "status", "lane_id", "speed", "volume"])
        print(df.head(10))  # 印前 10 筆
        print(f"✅ 共解析 {len(df)} 筆")
    else:
        print("⚠️ 沒有解析到任何資料")


if __name__ == "__main__":
    if len(sys.argv) == 3:
        date, fname = sys.argv[1], sys.argv[2]
        debug_file(date, fname)
    else:
        print("用法：")
        print("  python debug_single_file.py 20240101 VDLive_0735.xml.gz")
