# eTag 專案說明文件

本專案用於 **下載、檢查、解析及統計高速公路 eTag 車流資料**，並輸出整理後的 CSV 供後續分析與視覺化使用。

---

## 1. `etag_download.py`
**用途**  
批次下載指定日期範圍內的 eTag XML 資料（5 分鐘一筆），並自動解壓縮。支援斷點續抓與下載失敗重試，並可選擇壓縮成 ZIP 備份。

**主要功能**
- 設定下載日期範圍與時段（`hours_range`）
- 下載 `.xml.gz` → 解壓成 `.xml`
- 已存在的檔案自動跳過
- 指數回退重試機制
- 自動補抓舊的失敗清單
- 可將一週資料打包成 ZIP

**會產生的資料**
- `etag_data/etag_YYYYMMDD/*.xml` — 原始 eTag XML 檔案
- `failed_files.csv` — 下載失敗的 URL 清單
- `etag_week.zip` — 打包的一週 XML 檔案（選擇性）

**執行範例**
```bash 
python etag_download.py
1. 下載資料
   └─ 執行 etag_download.py
       ↓
       產生 etag_data/etag_YYYYMMDD/*.xml（原始 XML 檔案）
       產生 failed_files.csv（下載失敗的 URL）
       （選擇性）產生 etag_week.zip（一週壓縮檔）

2. 檢查資料完整性
   └─ 執行 check.py
       ↓
       在終端機顯示缺檔 & 0 Bytes 檔案的檢查結果

3. 解析與統計
   └─ 執行 etag_data_to_csv.py
       ↓
       產生 eTag_加權統計_一週.csv（整理後統計結果）
       產生 missing_files.csv（缺檔清單）
       產生 parse_errors.log（解析失敗紀錄）
      （程式只有在有錯誤/缺檔時才會生成這兩個檔案）
