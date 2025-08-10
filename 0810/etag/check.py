# 檢查etag資料完整性

import os 

# ===== 設定 =====
data_folder = "etag_data"      # 存放一週資料的主資料夾
hours_range = range(0, 24)     # 如果只抓 06:00–21:55 改成 range(6, 22)

# 預期檔名列表
expected_files = [f"ETagPairLive_{h:02d}{m:02d}.xml" 
                  for h in hours_range for m in range(0, 60, 5)]

# 開始檢查
for day_folder in sorted(os.listdir(data_folder)):
    folder_path = os.path.join(data_folder, day_folder)
    if not os.path.isdir(folder_path):
        continue
    
    files = [f for f in os.listdir(folder_path) if f.endswith(".xml")]
    file_set = set(files)
    zero_files = [f for f in files if os.path.getsize(os.path.join(folder_path, f)) == 0]
    missing_files = sorted(set(expected_files) - file_set)
    
    print(f"📅 {day_folder}:")
    print(f"  檔案數: {len(files)}/{len(expected_files)}")
    print(f"  0 Bytes 檔案數: {len(zero_files)}")
    if missing_files:
        print(f"  缺少 {len(missing_files)} 檔案 → 範例: {missing_files[:5]}")
    else:
        print("  ✅ 時間點完整")
    print()
