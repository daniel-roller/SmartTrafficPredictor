# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import warnings

warnings.filterwarnings("ignore")
os.environ["JOBLIB_MULTIPROCESSING"] = "0"
os.makedirs("../results", exist_ok=True)

# === 定義模型 ===
models = {
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(),
    "KNN": KNeighborsClassifier(),
    "XGBoost": XGBClassifier(eval_metric='logloss'),
    "LightGBM": LGBMClassifier(verbose=-1, force_col_wise=True, n_jobs=1),
    "CatBoost": CatBoostClassifier(verbose=0),
}

# === 結果儲存 ===
results = []
report_texts = []

# === 評估函式 ===
def evaluate_and_log(model, name, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    results.append({"Model": name, "Accuracy": acc, "F1 Score": f1})
    report = classification_report(y_test, y_pred)
    report_texts.append(f"==== {name} ====" + "\n" + report + "\n")
    print(f"{name} 完成 ✓")
    print(report)
    print("=" * 60)

# === 載入與預處理資料集 ===
datasets = {
    "Wine Quality": pd.read_csv("../data/wine_processed.csv"),
    "Heart Disease": pd.read_csv("../data/heart_processed.csv")
}

# === 訓練與評估所有模型 ===
for dataset_name, df in datasets.items():
    X = df.drop("target", axis=1)
    y = df["target"]

    print(f"\n📊 資料集：{dataset_name}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    for name, model in models.items():
        if name in ["SVM", "KNN"]:
            evaluate_and_log(model, f"{dataset_name} - {name}", X_train_scaled, X_test_scaled, y_train, y_test)
        else:
            evaluate_and_log(model, f"{dataset_name} - {name}", X_train, X_test, y_train, y_test)

# === 匯出結果 ===
results_df = pd.DataFrame(results)
results_df[["Dataset", "Model"]] = results_df["Model"].str.extract(r'(.*) - (.*)')
results_df = results_df.sort_values(by=["Dataset", "F1 Score"], ascending=[True, False])
results_df.to_csv("../results/comparison_results.csv", index=False, encoding="utf-8-sig")

with open("../results/classification_report.txt", "w", encoding="utf-8") as f:
    f.writelines(report_texts)

print("\n📊 模型效能總比較：")
print(results_df.to_string(index=False))
print("\n✅ 所有結果已儲存至 'results/' 資料夾")
