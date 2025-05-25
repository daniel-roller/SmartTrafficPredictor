# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
import warnings
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Microsoft JhengHei'
plt.rcParams['axes.unicode_minus'] = False

warnings.filterwarnings("ignore")
os.environ["JOBLIB_MULTIPROCESSING"] = "0"
os.makedirs("../results", exist_ok=True)
os.makedirs("../results/confusion matrix", exist_ok=True)

# ========== 定義模型與參數 ==========
models = {
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(),
    "KNN": KNeighborsClassifier(),
    "XGBoost": XGBClassifier(eval_metric='logloss'),
    "LightGBM": LGBMClassifier(verbose=-1, force_col_wise=True, n_jobs=1),
    "CatBoost": CatBoostClassifier(save_snapshot=False, logging_level="Silent")
}

param_grids = {
    "Decision Tree": {"max_depth": [3, 5, 10, None], "min_samples_split": [2, 5, 10]},
    "Random Forest": {"n_estimators": [50, 100], "max_depth": [5, 10, None]},
    "SVM": {"C": [0.1, 1, 10], "kernel": ['linear', 'rbf']},
    "KNN": {"n_neighbors": [3, 5, 7], "weights": ['uniform', 'distance']},
    "XGBoost": {"n_estimators": [50, 100], "max_depth": [3, 5], "learning_rate": [0.1, 0.01]},
    "LightGBM": {"n_estimators": [50, 100], "max_depth": [-1, 5, 10], "learning_rate": [0.1, 0.01]},
    "CatBoost": {"depth": [4, 6], "learning_rate": [0.1, 0.01]}
}

results = []
report_texts = []

# ========== 評估函數（含交叉驗證 + 混淆矩陣預測） ==========
def evaluate_model(model, name, X, y, param_grid, X_test_full, y_test_full, dataset_key):
    if param_grid:
        try:
            grid = GridSearchCV(model, param_grid, cv=3, scoring='f1', n_jobs=-1)
            grid.fit(X, y)
            model = grid.best_estimator_
            print(f"✓ {name} 使用最佳參數: {grid.best_params_}")
        except Exception as e:
            print(f"⚠ {name} GridSearch 失敗，使用預設參數: {e}")

    # Cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    acc = cross_val_score(model, X, y, cv=skf, scoring="accuracy").mean()
    f1 = cross_val_score(model, X, y, cv=skf, scoring="f1").mean()
    results.append({"Model": f"{dataset_key} - {name}", "Accuracy": acc, "F1 Score": f1})
    print(f"{name} 完成 ✓\n平均 Accuracy: {acc:.4f}, 平均 F1 Score: {f1:.4f}")
    print("=" * 60)

    # Confusion Matrix 預測與報表
    model.fit(X, y)
    y_pred = model.predict(X_test_full)
    report = classification_report(y_test_full, y_pred)
    report_texts.append(f"==== {dataset_key} - {name} ====\n{report}\n")

    # 輸出預測結果
    records = [{"Model": name, "y_test": yt, "y_pred": yp} for yt, yp in zip(y_test_full, y_pred)]
    out_file = dataset_key.lower().replace(" ", "_") + "_predictions.csv"
    pd.DataFrame(records).to_csv(f"../results/confusion matrix/{out_file}", index=False)

# ========== 載入與處理資料 ==========
datasets = {
    "Wine Quality": pd.read_csv("../data/wine_processed.csv"),
    "Heart Disease": pd.read_csv("../data/heart_processed.csv"),
    "Breast Cancer": pd.read_csv("../data/breast_cancer_processed.csv"),
    "Online Retail": pd.read_csv("../data/online_retail_processed.csv")
}

use_gridsearch = False
use_smote = True

for dataset_key, df in datasets.items():
    print(f"\n📊 資料集：{dataset_key}")
    
    if dataset_key == "Online Retail":
        df = df.sample(n=3000, random_state=42)

    X = df.drop("target", axis=1)
    y = df["target"]

    # 保留一份 test set 專供混淆矩陣使用
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    if use_smote:
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        print(f"📌 已使用 SMOTE 過採樣：{dataset_key}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_full_scaled = scaler.fit_transform(X)  # 給 cross-val 用

    for name, model in models.items():
        param_grid = param_grids.get(name) if use_gridsearch else None
        if name in ["SVM", "KNN"]:
            evaluate_model(model, name, X_train_scaled, y_train, param_grid, X_test_scaled, y_test, dataset_key)
        else:
            evaluate_model(model, name, X_train, y_train, param_grid, X_test, y_test, dataset_key)

# ========== 儲存結果 ==========
results_df = pd.DataFrame(results)
results_df[["Dataset", "Model"]] = results_df["Model"].str.extract(r'(.*) - (.*)')
results_df = results_df.sort_values(by=["Dataset", "F1 Score"], ascending=[True, False])
results_df.to_csv("../results/comparison_results.csv", index=False, encoding="utf-8-sig")

with open("../results/classification_report.txt", "w", encoding="utf-8") as f:
    f.writelines(report_texts)

print("\n📊 模型效能總比較：")
print(results_df.to_string(index=False))
print("\n✅ 所有結果已儲存至 'results/' 資料夾")

# ========== 圖表繪製 ==========
from plot_results import plot_comparison_charts
plot_comparison_charts()
