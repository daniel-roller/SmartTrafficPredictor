## ✅ 專題目標
比較不同機器學習模型在多種資料集上的表現，建立標準流程，為後續深度學習奠基。

---

## 1️⃣ 資料準備

- 🔍 從 Kaggle（KGO）、UCI ML Repo 找 3~5 筆資料集（類型不同佳）
- 📊 任務類型包含：
  - 分類（如 Titanic、生存預測、Iris）
  - 回歸（如 California Housing、房價預測）
  - 分群（如 顧客聚類、交通流量）

---

## 2️⃣ 資料前處理（Preprocessing）

- 缺失值處理：填補或刪除
- 類別欄位轉換：One-hot encoding
- 數值標準化：StandardScaler / MinMaxScaler
- 特徵篩選（視情況）

---

## 3️⃣ 模型實作與測試（Scikit-learn）

### 分類模型
- SVM（linear / poly）
- Decision Tree
- Random Forest
- KNN
- XGBoost（需安裝）

### 回歸模型
- Linear / Ridge / Lasso Regression
- Random Forest Regressor
- XGBoost Regressor

### 分群模型
- KMeans
- DBSCAN
- GMM
- Hierarchical Clustering

---

## 4️⃣ 評估與結果輸出

- 分類：Accuracy、F1-score
- 回歸：MSE、R²
- 分群：視覺化 or Silhouette Score
- ➕ 可加平均值、標準差作綜合比較

---

## 5️⃣ 製作比較表格（樣式示意）

| Dataset / Model | SVM | Decision Tree | Random Forest | ... |
|------------------|-----|----------------|----------------|-----|
| Iris             | 0.95| 0.93           | 0.96           |     |
| Titanic          | ... | ...            | ...            |     |
| 平均值           |     |                |                |     |
| 標準差           |     |                |                |     |

---

## 6️⃣ 超參數調整（進階）

- SVM：`kernel`, `C`, `gamma`
- Decision Tree：`max_depth`, `min_samples_split`
- Random Forest：`n_estimators`, `max_features`
- XGBoost：`learning_rate`, `n_estimators`, `max_depth`

---

## 🔜 7️⃣ 深度學習（未來延伸）

- DNN、CNN、Transformer（視任務選擇）

---

## 📌 小提醒
- 可參考 Kaggle 的 Titanic 範例學資料前處理方式
- 可從 default 參數開始跑，再挑選表現較好的模型做調整
- 記得把「大象」跟「長頸鹿」也放進去預測（教授 hint 😆）

教授原話精簡版:

• 你們可以去KGO去找 Machine Learning Benchmark Data。

• 你們先找三五個 Data，型態最好不一樣。

• 你們可以去看別人怎麼解，參考 KAGGLE 上的 Dataset。

• 你們最好還是把大象放進去、長頸鹿放進去，也要預測。

• 你們先用 default 參數跑各模型比較結果，再選一個效果較好的進行超參數調整。

• 你們要找分類、回歸、分群三種型態的資料，跑不同模型看看效果。

• 你們去看 Titanic 生存預測範例，那些處理資料方式滿值得參考的。