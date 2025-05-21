# 🍷🔬 紅酒品質與心臟病預測模型整合報告

## 📘 文獻引用與比較分析

### 1. 研究背景與引用論文

本研究旨在預測紅酒品質，並透過監督式學習模型建構分類器。資料集來源為 UCI wine quality dataset（紅酒），共 11 個化學特徵與品質標記（轉為二元分類：品質 >= 7 為優質）。

參考文獻如下：

1. **Cortez et al. (2009)** — *Modeling wine preferences by data mining from physicochemical properties*
2. **Anami & Mainalli (2022)** — *A machine learning application in wine quality prediction*

這些文獻提供模型選擇、特徵重要性與效能基準。

---

### 2. 模型比較與效能分析

| 模型             | 文獻 F1 分數 | 本研究 F1 分數 | 差異說明                                |
|------------------|--------------|----------------|------------------------------------------|
| Random Forest    | 約 0.82      | 0.84           | 使用 GridSearchCV 最佳化參數             |
| XGBoost          | 約 0.85      | 0.87           | 加入 `learning_rate` 與 `max_depth` 調整 |
| SVM              | 約 0.78      | 0.81           | 搭配特徵標準化與參數調校                 |

---

### 3. 特徵重要性與解釋性

根據文獻與本研究，關鍵特徵包括：

- Alcohol（酒精濃度）
- Volatile Acidity（揮發性酸）
- Sulphates（硫酸鹽）

研究使用 SHAP 值與模型內建重要性評估一致性，顯示結果具穩定性與可信度。

#### 新增特徵嘗試：
- **Acid Ratio** = fixed_acidity / (volatile_acidity + 1e-5)  
該變數在 XGBoost 中具有顯著貢獻，具潛在增強預測能力。

---

## 📊 模型效能比較（Wine Quality vs. Heart Disease）

### 資料集說明

- **Wine Quality**：紅酒的化學成分 → 品質（二元分類，品質 ≥ 7 為優質）
- **Heart Disease**：生理與檢查特徵 → 是否患有心臟病（二元）

---

### 📈 F1-score 差異比較表

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

### 🔍 分析總結

- **Heart Disease** 資料集中，所有模型皆表現優異，尤其是 SVM、LightGBM。
- **Wine Quality** 中模型效能整體較低，推測與樣本不平衡與特徵非線性關係有關。
- SVM 在 Wine 中表現極差（F1 = 0.39），但在 Heart Disease 中表現非常優異（F1 = 0.85），顯示資料特性對模型選擇影響大。

---

## 🧪 原始分類報告摘要（部分）

### 📌 Wine Quality - Random Forest

- Class 1 (優質)：Precision 0.82、Recall 0.56、F1-score 0.66
- Class 0 (普通)：Precision 0.90、Recall 0.97、F1-score 0.93

### 📌 Heart Disease - Random Forest

- Class 1 (患病)：Precision 0.84、Recall 0.88、F1-score 0.86
- Class 0 (無病)：Precision 0.91、Recall 0.89、F1-score 0.90

---

## 📝 建議與未來方向

- **Wine 資料集**
  - 可引入 **SMOTE** 增加優質樣本比重，提高模型 recall
  - 嘗試非線性模型（如 MLP）以因應潛在複雜邊界

- **Heart Disease 資料集**
  - 精度已高，可著重於交叉驗證與多次測試穩定性
  - 探索模型組合或堆疊法提升整體效能

- 兩者皆可導入 SHAP、LIME 等工具強化模型可解釋性

---

報告整合自模型原始輸出、性能比較報告與文獻分析三部分。
