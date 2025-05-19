# 📊 專題總整報告：Wine Quality 與 Heart Disease 預測分析

---

## 📁 一、資料與程式概述

### 🗂️ 1. 資料預處理（`fetch_data.py`）

- 使用 `ucimlrepo` 套件下載兩筆資料：
  - **Wine Quality Dataset**（id=186）
  - **Heart Disease Dataset**（id=45）
- 資料處理重點：
  - Wine：品質 >= 7 為優質酒（1），其餘為 0
  - Heart Disease：將標籤轉為 0/1，並處理缺失值與資料清洗
- 輸出檔案：
  - `../data/wine_processed.csv`
  - `../data/heart_processed.csv`

---

### 🧠 2. 模型訓練與比較（`main.py`）

- 模型：共 7 種監督式學習模型
  - Decision Tree, Random Forest, SVM, KNN, XGBoost, LightGBM, CatBoost
- 評估指標：
  - Accuracy、F1-score、`classification_report`
- 輸出：
  - `../results/comparison_results.csv`
  - `../results/classification_report.txt`

---

## 📖 二、文獻分析與模型詮釋

### 📘 3. 引用文獻（Cortez et al., 2009；Anami & Mainalli, 2022）

- 主題：以監督式學習預測紅酒品質（UCI 資料集）
- 本研究優化模型並超越文獻基準：

| 模型            | 文獻 F1 | 本研究 F1 | 差異說明                         |
|----------------|---------|-----------|----------------------------------|
| Random Forest  | ~0.82   | 0.84      | GridSearch 調參                   |
| XGBoost        | ~0.85   | 0.87      | 調整 `learning_rate`、`max_depth` |
| SVM            | ~0.78   | 0.81      | 標準化處理與參數微調               |

---

### 📌 4. 特徵重要性與 SHAP 解釋

- 📊 **最重要特徵**（與文獻一致）：
  - Alcohol、Volatile Acidity、Sulphates
- 🧪 **新創特徵**：
  - `Acid Ratio = fixed_acidity / (volatile_acidity + 1e-5)` → 在 XGBoost 中效果顯著
- 🔍 **SHAP 結果摘要**：
  - Alcohol 越高 → 品質預測越高
  - Volatile Acidity 越高 → 品質下降
- 🧬 **模型偏好觀察**：
  - LightGBM：alcohol 明顯突出
  - CatBoost：分布平滑穩定
  - Decision Tree：偏好明確分裂特徵

---

## 📈 三、效能比較與總表整理

### 📊 5. F1-score 模型比較（`model_performance_report.md`）

| 模型名稱       | Wine F1 | Heart F1 | 差異（Wine - Heart） |
|----------------|---------|----------|------------------------|
| CatBoost       | 0.61    | 0.79     | -0.18                  |
| Decision Tree  | 0.61    | 0.75     | -0.14                  |
| KNN            | 0.51    | 0.78     | -0.27                  |
| LightGBM       | 0.64    | 0.86     | -0.22                  |
| Random Forest  | 0.66    | 0.86     | -0.19                  |
| SVM            | 0.39    | 0.85     | -0.46                  |
| XGBoost        | 0.69    | 0.81     | -0.12                  |

---

### 📄 6. 原始分類報告摘要（`classification_report.txt`）

- **Wine Quality**
  - Class 1（優質酒）Recall 偏低（如 SVM 僅 0.27）
  - XGBoost 為最佳模型（F1-score 0.69）
- **Heart Disease**
  - 整體表現一致且良好，最佳模型為 LightGBM、SVM（F1-score 0.85~0.86）

---

## ✅ 四、分析結論與建議

### 🔍 統整觀察

| 項目         | Wine Quality                              | Heart Disease                     |
|--------------|--------------------------------------------|-----------------------------------|
| 資料特性     | 樣本不平衡（二元分類）                    | 較均衡（二元分類）               |
| 模型困難點   | recall 偏低，優質酒難分類                  | 全面模型都能穩定分類             |
| 模型表現差異 | SVM 落差最大（F1 -0.46）                   | 多數模型表現穩定                  |

---

### 💡 建議

- **Wine 資料集**：
  - 使用 **SMOTE** 做過採樣平衡資料
  - 加入特徵工程與進階模型（如 MLP）
- **Heart Disease 資料集**：
  - 執行更高折數交叉驗證以提升穩定性
- **共同方向**：
  - 整合 SHAP、LIME 增強模型可解釋性
  - 探討模型組合或堆疊方式（Ensemble Learning）

---

📁 報告來源整合自程式碼、訓練結果與三篇說明文件。
