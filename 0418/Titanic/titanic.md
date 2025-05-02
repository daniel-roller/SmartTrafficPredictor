# 💚 Titanic Machine Learning Project - README

---

## 🔎 工作目標

將 Titanic 資料集用於分類預測，比較多個 Machine Learning 模型，評估效果，選最好的模型，並用圖表可視化，後續生成或交付成果。


---

## 📚 目錄結構 (tree .)

```plaintext
TitanicProject/
├── data/
│   ├— train.csv
│   └— test.csv
├── model/
│   └— titanic_best_model.pkl
├── output/
│   ├— titanic_prediction_result.csv
│   └— submission.csv
├── figures/
│   ├— model_accuracy.png
│   ├— model_f1score.png
│   ├— confusion_matrix.png
│   └— feature_importance.png
├── scripts/
│   ├— preprocessing.py
│   ├— train_and_plot.py
│   ├— load_and_predict.py
│   └— train_and_compare.py
├── README.md
```


---

## 🔢 環境要求

```bash
pip install pandas scikit-learn matplotlib seaborn joblib
```

---

## ⚡ 執行流程

### 1. 執行基礎版模型試證

🔹 執行基礎模型試證:

```bash
python scripts/train_and_plot.py
```

會產生：
- 比較模型測試正確率 (model_accuracy.png)
- 存成最好模型 titanic_best_model.pkl


### 2. 載入模型預測結果

```bash
python scripts/load_and_predict.py
```

會產生：
- output/titanic_prediction_result.csv
- output/submission.csv (Kaggle 上傳用)


### 3. 執行完整比較分析版

```bash
python scripts/train_and_compare.py
```

會生成：
- 比較模型 Accuracy 與 F1-score
- 示意圖：model_accuracy.png, model_f1score.png
- 混淆矩陣圖 confusion_matrix.png
- 特徵重要性圖 feature_importance.png (Decision Tree, Random Forest)


---

## ✨ 你會看到什麼？

- 🔢 Titanic 模型比較結果
- 📊 Accuracy 與 F1-score 圖
- 💛 最好的預測模型
- 🖌 混淆矩陣分析
- 🌐 特徵重要性分析


---

## 👉 擔心點

- 必須確保 `/data/` `/model/` `/figures/` `/output/` 路徑存在
- 圖片保存路徑是相對路徑（../）
- 預測模型不支援 feature_importance 的，會說明


---

## 🎉 完成目標

- 完整 Titanic Classification Project
- 網站標準的網頁組織
- 清楚說明說明
- 最好模型保存
- 可直接供給教授、考核或給組員使用


---

🚀 **Good Luck with Your Project!!** 🚀

