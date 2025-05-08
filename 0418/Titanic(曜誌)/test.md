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

## ⚙️ 常見模型參數說明與用途（+ 圖解表）

以下是模型中常見的參數功能整理：

| 模型 | 參數名稱 | 控制意義 | 適合調整的情況 |
|------|-----------|-----------|------------------|
| Decision Tree | `max_depth` | 限制樹的深度，防止過度記憶（過擬合） | 資料複雜時避免太細分 |
| Decision Tree | `min_samples_split` | 節點下去之前至少要有幾筆資料 | 避免分到沒意義 |
| Random Forest | `n_estimators` | 森林裡樹的數量 | 越多越穩定，但會變慢 |
| SVM | `C` | 容錯程度（大 C 越嚴格） | 想讓分類邊界更準時調整 |
| SVM | `gamma` | 每個資料點影響力 | 越大越容易過擬合 |
| Logistic Regression | `max_iter` | 最多訓練次數 | 避免模型還沒學好就停下來 |

📌 這些參數可以用 GridSearchCV 等方式自動找最佳值，也可以根據「哪個模型錯誤多」去調整對應參數。

---

## 📈 使用的分析方法與評估指標

| 方法 | 說明 |
|------|------|
| Accuracy | 預測正確的比例（基本指標） |
| F1-score | 同時考慮 Precision 與 Recall，適合資料不平衡 |
| Confusion Matrix | 分析預測錯在哪一類（誤殺或漏救） |
| Feature Importance | 解釋模型做出判斷時依賴的特徵（只有 DT/RF 有） |

---

## ✅ 如何判斷「模型好不好」？

很多人會困惑：「我到底怎麼知道這個模型做得好不好？」

✅ 最簡單的做法：
1. 先看 Accuracy 是否達到 baseline（例如 75%、80%）
2. 再看 F1-score 有沒有補足分類不平衡的問題
3. 看 Confusion Matrix：
   - 是不是一直漏掉生還者（FN 多）
   - 還是亂猜太多（FP 多）

📌 如果你發現分類某一類錯太多，就可以：
- 調 `C`（讓分類更嚴格或更寬容）
- 降低 `max_depth`（避免記太細）
- 增加 `n_estimators`（讓隨機森林更穩）

✅ 如果你還是抓不到感覺，你也可以「直接寫一個新的模型測試程式」，
只測試某一組參數，看看效果是不是更好。

我們可以再另外新增一個 `train_and_tune.py`，專門給你做：
- 測試不同 `max_depth`、`C`、`gamma`
- 自動列出每組參數的 Accuracy / F1-score

這樣你就能慢慢看出來：「參數怎麼調會讓結果變好」。

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

📌 如果還有不懂的地方，例如：
- 想實作 GridSearch 調參
- 想從圖表看出哪裡可以改進
- 想比較更多新模型（像 XGBoost）

我們可以再新增實驗檔案幫你一起完成！
