# 🧠 Titanic 生存預測專案說明文件

---

## 🎯 專案目標

本專案以 Titanic 生存預測為任務，透過多種機器學習模型對乘客是否生還進行分類，  
包含完整的資料處理、模型訓練、模型比較、模型解釋與預測成果輸出。

---

## 🔁 整體分析流程

> ✅ 你的理解是對的：  
> 我們會先分析資料 → 建立與比較多個模型 → 找出最佳模型 → 用它來做預測與評估 → 輸出成果（如 submission.csv）

分析流程如下：

1. **資料前處理**（處理缺失值、刪除欄位、標準化）
2. **模型訓練與比較**（比較不同模型：Decision Tree、Random Forest、SVM、Logistic Regression）
3. **模型評估與圖表分析**（Accuracy、F1-score、Confusion Matrix、Feature Importance）
4. **模型儲存與預測應用**（儲存最佳模型，對 test.csv 預測並輸出 Kaggle 格式）
5. **圖表報告輸出與解釋**（整合資料與歷史背景解釋模型結果）

---

## 📂 各檔案功能說明

### 1️⃣ `preprocessing.py`  
📦 模組化的資料前處理功能。

- 補齊缺失值（`Age`、`Fare`、`Embarked`）
- 移除無用欄位（`Cabin`、`Ticket`、`Name`、`PassengerId`）
- 保留 `PassengerId` 給 `submission.csv` 使用
- 封裝為函式：`preprocess_titanic(filepath, is_train=True/False)`

🔁 其他檔案都會呼叫這個函式，避免重複寫前處理。

---

### 2️⃣ `train_and_plot.py`  
🔍 初版簡易模型比較與訓練工具。

- 訓練三種模型（Decision Tree, Random Forest, SVM）
- 使用 Accuracy 為指標，找出表現最好的模型
- 畫出模型比較圖與特徵重要性圖
- 儲存最佳模型到 `model/`

📌 適合用來做「先跑跑看、確認模型是否運作正常」的版本。

---

### 3️⃣ `train_and_compare.py`  
🎓 正式分析用的核心訓練與比較工具。

- 使用前處理模組處理資料
- 訓練四種模型（DT、RF、SVM-RBF、Logistic Regression），皆有設定超參數
- 使用 Accuracy 與 F1-score 同時評估表現
- 輸出以下圖表：

| 圖表檔名 | 說明 |
|---------|------|
| `model_accuracy.png` | 模型準確率比較 |
| `model_f1score.png` | 模型 F1-score 比較 |
| `confusion_matrix.png` | 模型混淆矩陣分析圖 |
| `feature_importance.png` | 模型特徵重要性圖（如支援） |

- 儲存最佳模型成 `.pkl`，供預測時載入使用

📌 是報告、交作業或發表時應該引用的主要程式。

---

### 4️⃣ `load_and_predict.py`  
🔮 用於模型應用與預測。

- 載入訓練後儲存的最佳模型 `.pkl`
- 對 `train.csv` 做預測，輸出成 `titanic_prediction_result.csv`
- 對 `test.csv` 做預測，產出符合 Kaggle 的 `submission.csv`

📌 可直接產出實際可用的結果提交到 Kaggle 評分。

---

## 📈 使用的分析方法與評估指標

| 方法 | 說明 |
|------|------|
| Accuracy | 預測正確的比例（基本指標） |
| F1-score | 同時考慮 Precision 與 Recall，適合資料不平衡 |
| Confusion Matrix | 分析預測錯在哪一類（誤殺或漏救） |
| Feature Importance | 解釋模型做出判斷時依賴的特徵（只有 DT/RF 有） |

---

## ✅ 評估流程是否正確？

是的！整個流程是完整正確的：

1. 前處理一致性好（用模組統一處理）
2. 模型訓練後有保留 `.pkl` 可重複使用
3. 有比較多模型，並用合理指標評估
4. 預測結果有存成 `.csv` 並區分訓練預測 vs 測試預測
5. 分析圖完整，可直接進報告使用

---

## 🧠 社會背景補充（資料與模型吻合）

- 女性與小孩優先 → 性別欄位變成最重要特徵
- 一等艙比較容易逃生 → Pclass 與 Fare 也變成關鍵特徵
- 模型學會這些分界 → Feature Importance 圖能明顯呈現

---

## 📦 輸出成果總整理

| 檔案 / 圖表 | 說明 |
|-------------|------|
| `model_accuracy.png` | 各模型準確率比較圖 |
| `model_f1score.png` | 各模型分類品質比較圖 |
| `confusion_matrix.png` | 最佳模型混淆矩陣（預測錯在哪） |
| `feature_importance.png` | 最佳模型依據哪些欄位判斷 |
| `titanic_best_model.pkl` | 訓練後模型（可重複使用） |
| `titanic_prediction_result.csv` | 預測 train.csv 結果 |
| `submission.csv` | 預測 test.csv 結果，可交 Kaggle |

---

## 🏁 總結

這個專案設計邏輯清楚、方法完整、可重現性高，  
分析過程中不只做出預測，也能從模型中解釋出資料背後的意義，  
是一份符合 Kaggle 賽事與報告型任務雙標準的專案架構！

