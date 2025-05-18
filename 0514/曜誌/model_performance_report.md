
# 📊 模型效能比較報告（Wine Quality vs. Heart Disease）

本報告整合兩筆資料集的模型預測效能，以 `F1-score` 為主軸進行比較。資料與結果由你提供的模型輸出彙整。

---

## 📁 資料集說明

1. **Wine Quality**：化學特徵 → 品質（品質 ≥ 7 為優質）
2. **Heart Disease**：檢測特徵 → 是否患病（0/1）

---

## 📈 F1-score 差異比較

| 模型名稱 | Wine F1 | Heart F1 | 差異（Wine - Heart） |
|----------|---------|----------|------------------------|
| CatBoost | 0.61 | 0.79 | -0.18 |
| Decision Tree | 0.61 | 0.75 | -0.14 |
| KNN | 0.51 | 0.78 | -0.27 |
| LightGBM | 0.64 | 0.86 | -0.22 |
| Random Forest | 0.66 | 0.86 | -0.19 |
| SVM | 0.39 | 0.85 | -0.46 |
| XGBoost | 0.69 | 0.81 | -0.12 |

---

## 🔍 分析總結

- 在 Heart Disease 中，**SVM** 和 **LightGBM** 取得了最高 F1 分數（>0.85）
- 在 Wine Quality 中，**XGBoost** 和 **Random Forest** 雖然整體準確率高，但 F1 分數落差較明顯，顯示品質預測更不平衡
- 所有模型在 Wine 資料中 F1 分數皆低於在 Heart Disease 中的分數（差異最大的是 SVM）

---

## 📝 建議

- 對 Wine Quality 可採取 SMOTE 過採樣提升 recall 與 minority class 表現
- Heart Disease 可使用交叉驗證提升穩定性

---

報告由系統自動根據輸入資料生成。
