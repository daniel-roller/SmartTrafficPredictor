# tune_heart_classification.py
# 調整 Heart Disease 分類模型參數 + SMOTE + SelectKBest + LightGBM + CatBoost

import pandas as pd
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import f1_score, accuracy_score
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from preprocess_heart import preprocess_heart
import joblib
import os

# === 1. 載入資料 ===
X, y = preprocess_heart("../data/heart.csv")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === 2. 建立前處理器 ===
num_features = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]
cat_features = ["Sex", "ChestPainType", "FastingBS", "RestingECG", "ExerciseAngina", "ST_Slope"]

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_features),
    ("cat", OneHotEncoder(drop="first"), cat_features)
])

# === 3. 定義模型與參數空間 ===
model_configs = {
    "XGBoost": {
        "model": XGBClassifier(eval_metric='logloss'),
        "params": {
            "select__k": [5, 7, 10],
            "model__n_estimators": [50, 100],
            "model__max_depth": [3, 5],
            "model__learning_rate": [0.05, 0.1]
        }
    },
    "KNN": {
        "model": KNeighborsClassifier(),
        "params": {
            "select__k": [5, 7, 10],
            "model__n_neighbors": [3, 5, 7],
            "model__weights": ["uniform", "distance"]
        }
    },
    "CatBoost": {
        "model": CatBoostClassifier(verbose=0, allow_writing_files=False, task_type="CPU"),
        "params": {
            "select__k": [10],
            "model__iterations": [50],
            "model__depth": [5],
            "model__learning_rate": [0.1]
        }
    }

}

# === 4. 執行 GridSearchCV 搜尋最佳參數 ===
best_models = {}
results = {}
for name, cfg in model_configs.items():
    print(f"🔍 搜尋中：{name}")
    pipe = Pipeline([
        ("pre", preprocessor),
        ("smote", SMOTE(random_state=42)),
        ("select", SelectKBest(score_func=f_classif)),
        ("model", cfg["model"])
    ])

    grid = GridSearchCV(pipe, cfg["params"], cv=5, scoring="f1", n_jobs=-1)
    grid.fit(X_train, y_train)

    best_models[name] = grid.best_estimator_
    y_pred = grid.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    results[name] = {
        "accuracy": acc,
        "f1": f1,
        "model": grid.best_estimator_
    }

    print(f"✅ {name} 最佳參數：{grid.best_params_}")
    print(f"➡️ 測試集 F1-score = {f1:.4f}, Accuracy = {acc:.4f}\\n")

# === 5. 儲存 F1-score 最佳者 ===
best_name = max(best_models, key=lambda n: f1_score(y_test, best_models[n].predict(X_test)))
best_model = best_models[best_name]

os.makedirs("../model", exist_ok=True)
joblib.dump(best_model, "../model/heart_best_tuned.pkl")
print(f"💾 已儲存最佳調參模型：{best_name} 至 ../model/heart_best_tuned.pkl")
# === 6. 模型評估圖表與混淆矩陣 ===
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# 建立資料夾
os.makedirs("../figures", exist_ok=True)

# 比較圖（Accuracy）
plt.figure(figsize=(10, 5))
plt.bar(results.keys(), [r["accuracy"] for r in results.values()], color="skyblue")
plt.xticks(rotation=15)
plt.ylabel("Accuracy")
plt.title("Heart Model Accuracy Comparison")
plt.tight_layout()
plt.savefig("../figures/heart_accuracy.png")

# 比較圖（F1-score）
plt.figure(figsize=(10, 5))
plt.bar(results.keys(), [r["f1"] for r in results.values()], color="salmon")
plt.xticks(rotation=15)
plt.ylabel("F1-score")
plt.title("Heart Model F1-score Comparison")
plt.tight_layout()
plt.savefig("../figures/heart_f1score.png")

print("📊 模型比較圖已儲存至 figures/heart_accuracy.png / heart_f1score.png")

# 混淆矩陣
y_pred = best_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
plt.title(f"Confusion Matrix ({best_name})")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("../figures/heart_confusion_matrix.png")

print("📊 混淆矩陣圖已儲存至 figures/heart_confusion_matrix.png")
