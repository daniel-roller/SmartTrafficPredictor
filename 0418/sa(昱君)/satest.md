# 📊 顧客資料分析專案 - 完整工作內容整理

## 🧱 1. 資料模擬與前處理

### ✅ 使用 `simulate_customer_data()` 產生模擬資料：
- **分類目標**：Segment（Segment A/B/C）
- **回歸目標**：PurchaseAmount（預測顧客購買金額）
- **數值特徵**：
  - `Age`：20–80 歲
  - `Income`：30,000–100,000 元
  - `PurchaseFreq`：1–20 次
  - `Membership`：是否為會員（二元）
- **其他欄位**：
  - `Gender`：隨機產生（Male/Female/Other）

### ✅ 特徵處理：
- 使用 `make_classification` 與 `make_regression` 模擬特徵與目標
- 所有特徵皆標準化為較真實的範圍
- 轉換 Membership 為 0/1
- PurchaseAmount 經修正為正值

---

## 🔍 2. 分群任務（Clustering）

### ▶ 分群流程：
1. 選擇特徵：`['Age', 'Income', 'PurchaseFreq']`
2. 標準化（`StandardScaler`）
3. 降維（`PCA` 2 維）
4. 分群模型：`KMeans`（測試 k = 2–5）
5. 分群後視覺化（使用 PCA 主成分）
6. 分析各群特徵均值

### ▶ 使用檔案：
- `sa.py`：封裝 `perform_clustering()`
- `main.py`：執行並儲存群聚視覺圖

---

## 🎯 3. 分類任務（Classification）

### ▶ 任務設定：
- **目標**：預測 `Segment`
- **特徵**：`['Age', 'Income', 'PurchaseFreq', 'Membership']`
- **模型**：`RandomForestClassifier`（100 顆樹、類別權重平衡）

### ▶ 執行步驟：
1. 資料分割（`train_test_split`, stratify）
2. 模型訓練與預測
3. 評估：
   - 分類報告（precision, recall, f1-score）
   - 混淆矩陣（視覺化）
   - 特徵重要性（bar plot）

### ▶ 輸出成果：
- CSV：`classification_report.csv`
- 圖片：混淆矩陣與特徵重要性圖
- 模型：儲存為 `classifier.pkl`

---

## 📈 4. 回歸任務（Regression）

### ▶ 任務設定：
- **目標**：預測 `PurchaseAmount`
- **特徵**：同分類任務
- **模型**：`GradientBoostingRegressor`

### ▶ 執行步驟：
1. 資料分割（訓練 / 測試集）
2. 模型訓練與預測
3. 評估：
   - RMSE（均方根誤差）
   - R²（決定係數）
   - 預測值 vs 實際值（散點圖）
   - 特徵重要性圖

### ▶ 輸出成果：
- CSV：`regression_metrics.csv`
- 圖片：回歸預測圖與特徵重要性圖
- 模型：儲存為 `regressor.pkl`

---

## 📦 5. 儲存與結果管理

### ▶ 主程式中（`main.py`）建立以下資料夾：
- `figures/`：儲存所有圖表
- `results/`：儲存報表（如 CSV）
- `models/`：儲存模型檔（`.pkl`）

---

## ✅ 總結

本專案涵蓋以下關鍵流程：

- 📐 模擬顧客資料與前處理
- 🌀 KMeans 分群 + PCA 降維視覺化
- 🎯 Segment 分類（Random Forest）
- 💰 PurchaseAmount 回歸（GBR）
- 📊 模型評估與視覺化
- 💾 模型與結果自動儲存（CSV + 圖片 + 模型）
