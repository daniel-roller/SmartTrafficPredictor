# Titanic 機器學習專案總結（六大主程式說明與執行流程）

本專案以 Kaggle Titanic 生還預測競賽為核心資料集，透過三大任務（分類、回歸、分群），完成多種模型比較、調參與預測輸出流程。

---

## 📁 一、六份主要 Python 檔案詳細說明

### 1. `preprocessing.py`（前處理模組）

* 功能：將原始資料清理、補值、轉換為模型可讀格式，統一訓練與測試資料使用方式。
* 特色：

  * 補齊 `Age`, `Fare`, `Embarked` 缺失值
  * 建立衍生特徵：

    * `Title`（從 `Name` 擷取）
    * `FamilySize`（家人數 = SibSp + Parch + 1）
    * `IsAlone`（是否一人）
    * `FarePerPerson`（平均票價）
  * 刪除無意義欄位：`Cabin`, `Ticket`, `Name`, `PassengerId`
* 回傳格式：訓練集回傳 dataframe，測試集同時回傳 `PassengerId`

---

### 2. `train_classification.py`（基本分類模型訓練與比較）

* 任務：預測 `Survived`（是否生還）
* 使用演算法：

  * Logistic Regression
  * Decision Tree（max\_depth = 3, 5）
  * Random Forest（n\_estimators = 50, 100; max\_depth = 5, 10）
  * SVM（C = 1, 10）
* 評估指標：Accuracy 與 F1-score，並找出 F1 最佳模型
* 輸出：

  * 儲存最佳模型為 `titanic_best_classification.pkl`
  * 產出模型比較圖與混淆矩陣圖

---

### 3. `tune_classification.py`（調參分類 + SMOTE + 特徵選擇）

* 核心強化：

  * 引入 SMOTE 處理不平衡資料（生還者少）
  * 加入 SelectKBest（f\_classif）挑選最佳特徵
  * 使用 GridSearchCV 進行超參數搜尋（cv=5）
* 使用演算法與參數範圍：

  * `XGBClassifier`

    * `n_estimators`: \[50, 100]
    * `max_depth`: \[3, 5]
    * `learning_rate`: \[0.05, 0.1]
    * `select__k`: \[5, 8, 10]
  * `KNeighborsClassifier`

    * `n_neighbors`: \[3, 5, 7]
    * `weights`: \["uniform", "distance"]
    * `select__k`: \[5, 8, 10]
* 評估指標：F1-score（更適合不平衡資料）
* 儲存最佳模型為：`titanic_best_tuned.pkl`

---

### 4. `train_regression.py`（預測票價的回歸模型）

* 任務：預測 `Fare`
* 使用演算法：

  * Linear Regression
  * Random Forest Regressor（n\_estimators = 50, 150; max\_depth = 5, 10）
  * SVR（C = 1, 10）
* 評估指標：

  * Mean Squared Error（MSE）
  * R² 分數
* 儲存最佳 MSE 模型為：`titanic_best_regressor.pkl`
* 輸出 MSE 比較圖

---

### 5. `train_clustering.py`（無監督分群 + 可視化）

* 任務：使用 KMeans 對乘客進行群分類
* 測試群數：K = 2, 3, 4, 5
* 選擇最佳群數指標：Silhouette Score
* 對最佳群數進行：

  * PCA 降維後繪製群分布圖
  * 分數比較折線圖
* 儲存最佳模型為：`titanic_best_cluster.pkl`

---

### 6. `load_and_predict.py`（統一模型載入與預測）

* 功能：

  * 讀入 `test.csv` 資料
  * 針對每個模型類型自動預測與輸出結果（分類 / 回歸 / 分群）
  * 輸出 `.csv` 至 `results/` 供上傳 Kaggle 或報告分析
* 支援模型與路徑對應：

  * `"classifier"` → `titanic_best_classification.pkl`
  * `"tuned_classifier"` → `titanic_best_tuned.pkl`
  * `"regressor"` → `titanic_best_regressor.pkl`
  * `"cluster"` → `titanic_best_cluster.pkl`

---

## 🧪 二、執行流程說明（從訓練到預測）

1. **準備資料與預處理**

   * 放入 `train.csv` 與 `test.csv` 至 `data/`
   * 執行主程式會自動呼叫 `preprocessing.py` 處理欄位與缺失值

2. **模型訓練階段**

   * `train_classification.py`：快速多模型比較
   * `tune_classification.py`：深入調參找最佳參數
   * `train_regression.py` / `train_clustering.py`：對應任務的延伸應用與分析

3. **模型預測與匯出**

   * 執行 `load_and_predict.py` → 自動處理 test 預測 + 輸出 `.csv`
   * 可直接上傳 Kaggle 驗證模型分數（已測試成功）

---

## 🦒🐘 三、「大象與長頸鹿」：老師的期待與呼應

> 「資料集不能都只有人，要有大象和長頸鹿」：意指資料應該具有『多樣性』與『泛化能力』。

### ✅ 我們目前做到：

* 任務類型多樣：分類、回歸、分群
* 分析面向完整：含特徵工程、SMOTE、調參、可視化
* 模型流程通用化，可支援未來擴充

### 🔶 尚未完全達成的部分：

* 資料來源仍為 Titanic（人類乘客）
* 若要涵蓋「大象長頸鹿」的概念，應進一步加入 **不同型態的資料集**，例如：

  * 醫療資料：心臟病預測（分類）
  * 房價預測：Kaggle House Price（回歸）
  * 顧客群分：Mall segmentation（分群）

### 🔁 建議策略：

* 為每個新資料集建立獨立資料夾與 `preprocessing_*.py`
* 使用相同架構的 `train_*.py` 模板，只改特徵與目標欄位
* 保留共用的 `load_and_predict.py` 或調整為可指定資料集版本

---

## ✅ 四、目前狀況總結

| 項目          | 狀態                       |
| ----------- | ------------------------ |
| 基本資料處理      | ✅ 完成                     |
| 三大任務架構      | ✅ 建立完成                   |
| 模型比較與可視化    | ✅ 有圖表輸出                  |
| 調參與模型最佳化    | ✅ 有 GridSearchCV + SMOTE |
| 多資料支援與彈性模組化 | 🔶 初具結構，尚未加入跨資料測試        |
| 呼應「多樣性資料任務」 | 🔶 可準備其他領域資料進行遷移測試       |

---

## 📎 補充建議與後續行動

* 加入 `submission_log.csv`，紀錄每次上傳結果與參數說明
* 設計資料集切換架構，讓每筆資料都能對應一套 train + preprocess 流程
* 透過繼承或模組化方式簡化不同資料集的切換與擴充

如需協助挑選合適資料集、建立新版的流程架構，請再告訴我。
