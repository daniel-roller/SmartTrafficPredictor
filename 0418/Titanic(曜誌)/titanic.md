# 🔍 Titanic 專案資料前處理統整說明

本文件說明專案中資料前處理的兩大階段：`preprocessing.py` 負責資料清理，`ColumnTransformer` 則負責特徵轉換。這樣的設計讓每個任務（分類、回歸、分群）都能重複使用統一的流程，同時確保模型可以正確訓練與預測。

---

## 🧹 第一階段：資料清理（由 `preprocessing.py` 處理）

此階段負責處理原始 CSV 資料，將其轉為乾淨、統一的 DataFrame。

### ✅ 功能整理

| 功能    | 說明                                                    |
| ----- | ----------------------------------------------------- |
| 缺值處理  | `Age` 補中位數，`Fare`（test）補中位數，`Embarked` 補眾數            |
| 欄位刪除  | `Cabin`, `Ticket`, `Name`, `PassengerId`（只保留一份）       |
| 欄位一致性 | 讓 train/test 擁有相同欄位結構（可交互使用）                          |
| 模式區分  | `is_train=True` 時保留 Survived；否則只保留 PassengerId + 特徵欄位 |

### 📦 範例輸出欄位（給 ColumnTransformer 使用）

```text
Pclass, Sex, Age, SibSp, Parch, Fare, Embarked
```

---

## 🏗️ 第二階段：特徵轉換（由 `ColumnTransformer` 處理）

這一階段是專門為了「機器學習模型能理解與學習」而設計的，主要將欄位轉成數值化格式並進行標準化。

### ✅ 處理內容

| 處理    | 工具                            | 說明                     |
| ----- | ----------------------------- | ---------------------- |
| 類別轉換  | `OneHotEncoder(drop='first')` | 例如 Sex=female 轉成 0/1   |
| 數值標準化 | `StandardScaler()`            | 讓 Age、Fare 等特徵轉成標準常態分布 |

```python
ColumnTransformer([
    ("num", StandardScaler(), ["Pclass", "Age", "SibSp", "Parch", "Fare"]),
    ("cat", OneHotEncoder(drop="first"), ["Sex", "Embarked"])
])
```

### ✅ 加入 Pipeline 的好處

| 優點           | 說明                            |
| ------------ | ----------------------------- |
| 可統一訓練流程      | `.fit()`、`.predict()` 都包含資料轉換 |
| 可被儲存為 `.pkl` | 模型包含完整前處理邏輯                   |
| 減少重複程式碼      | 每個任務不需再自己轉欄位                  |

---

## 🧠 統整比較表

| 功能       | `preprocessing.py` | `ColumnTransformer` |
| -------- | ------------------ | ------------------- |
| 缺值補齊     | ✅                  | ❌                   |
| 刪除無用欄位   | ✅                  | ❌                   |
| 資料格式轉數字  | ❌                  | ✅                   |
| 標準化數值特徵  | ❌                  | ✅                   |
| 輸出供模型訓練  | ✅                  | ✅（經由 pipeline）      |
| 可與模型打包儲存 | ❌                  | ✅                   |

---

## ✅ 結論

前處理分兩階段各司其職：

* `preprocessing.py` 負責基本清洗與欄位統一
* `ColumnTransformer` 負責模型能接受的數值轉換

這種結構讓整個專案具有：

* 模組化（可重複用）
* 一致性（train/test 處理邏輯一致）
* 可擴充（可以在不同模型中複用）

你現在的程式架構已經非常穩固，推薦保留這樣的設計！
