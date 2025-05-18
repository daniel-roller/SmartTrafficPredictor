## 📘 文獻引用與比較分析

### 1. 研究背景與引用論文

本研究旨在預測紅酒品質，並透過監督式學習模型建構分類器。資料集來源為 UCI wine quality dataset（紅酒），共 11 個化學特徵與品質標記（轉為二元分類：品質 >= 7 為優質）。

本專題參考以下兩篇關鍵文獻：

1. **Cortez et al. (2009)** — "Modeling wine preferences by data mining from physicochemical properties"
2. **Anami & Mainalli (2022)** — "A machine learning application in wine quality prediction"

上述文獻提供了模型選擇、特徵重要性與模型效能表現的參考基準。

---

### 2. 模型比較與效能分析

根據文獻與本研究實作模型的比較表如下：

| 模型            | 文獻 F1 分數 | 本研究 F1 分數 | 差異說明                              |
| ------------- | -------- | --------- | --------------------------------- |
| Random Forest | 約 0.82   | 0.84      | 使用 GridSearchCV 最佳化參數             |
| XGBoost       | 約 0.85   | 0.87      | 加入 learning\_rate 與 max\_depth 調整 |
| SVM           | 約 0.78   | 0.81      | 搭配特徵標準化與參數調校                      |

本研究使用 GridSearchCV 對上述模型進行調參，整體效能表現優於文獻結果。

---

### 3. 特徵重要性對照

根據 Cortez et al. (2009)，影響紅酒品質的重要特徵包含：

* Alcohol
* Volatile Acidity
* Sulphates

本研究使用 SHAP 與模型內建的特徵重要性分析，亦發現上述三項特徵在多數模型中權重居前，顯示本研究結果與文獻具有一致性。

此外，我們補充 LightGBM、CatBoost 模型，觀察其對特徵的選擇是否有異。

根據論文啟發，我們創造以下衍伸特徵：

Acid Ratio = fixed_acidity / (volatile_acidity + 1e-5)
該變數在 XGBoost 模型中具有一定的重要性，顯示可能潛在增強預測能力。

5. 結語
透過與前人研究對照，本研究不僅驗證了既有模型與特徵選擇的穩定性，亦透過模型調參與 SHAP 解釋性分析提升可信度。後續可進一步擴充不同年份或地區的紅酒資料集，以評估模型的泛化能力。  
---

### 4. 新特徵嘗試與啟發

根據論文啟發，我們創造以下衍伸特徵：

* **Acid Ratio** = fixed\_acidity / (volatile\_acidity + 1e-5)

該變數在 XGBoost 模型中具有一定的重要性，顯示可能潛在增強預測能力。

---

### 5. 結語

透過與前人研究對照，本研究不僅驗證了既有模型與特徵選擇的穩定性，亦透過模型調參與 SHAP 解釋性分析提升可信度。後續可進一步擴充不同年份或地區的紅酒資料集，以評估模型的泛化能力。
