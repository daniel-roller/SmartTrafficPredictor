# train_classification.py
# 分類任務（預測 Titanic 乘客是否生還），整合模型比較 + 多組參數設定

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import numpy as np
import os
from preprocessing import preprocess_titanic

# === 1. 載入前處理後的 Titanic 資料 ===
df = preprocess_titanic("../data/train.csv", is_train=True)
X = df[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]]
y = df["Survived"]

# === 2. 分割資料集 ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === 3. 建立通用前處理器 ===
num_features = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
cat_features = ["Sex", "Embarked"]
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_features),
    ("cat", OneHotEncoder(drop="first"), cat_features)
])

# === 4. 定義多模型 + 多參數組合 ===
models = {
    "DecisionTree": [
        ("depth3", DecisionTreeClassifier(max_depth=3, random_state=42)),
        ("depth5", DecisionTreeClassifier(max_depth=5, random_state=42))
    ],
    "RandomForest": [
        ("n50", RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)),
        ("n100", RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42))
    ],
    "SVM": [
        ("C1", SVC(C=1, gamma="auto")),
        ("C10", SVC(C=10, gamma="scale"))
    ],
    "LogisticRegression": [
        ("default", LogisticRegression(max_iter=1000))
    ]
}

# === 5. 模型訓練 + 評估（Accuracy / F1） ===
results = {}
for model_name, configs in models.items():
    for label, model in configs:
        tag = f"{model_name}_{label}"
        pipe = Pipeline([
            ("pre", preprocessor),
            ("model", model)
        ])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        results[tag] = {
            "accuracy": acc,
            "f1": f1,
            "model": pipe,
            "y_pred": y_pred
        }
        print(f"{tag} ➜ Accuracy = {acc:.4f}, F1 = {f1:.4f}")

# === 6. 找出最佳模型（以 F1 分數為主） ===
best_tag = max(results, key=lambda k: results[k]["f1"])
best_model = results[best_tag]["model"]
print(f"\n✅ Best model: {best_tag}")

# === 7. 儲存最佳模型 ===
os.makedirs("../model", exist_ok=True)
joblib.dump(best_model, "../model/titanic_best_classification.pkl")
print("✅ 模型已儲存為 model/titanic_best_classification.pkl")

# === 8. 繪製比較圖（Accuracy / F1） ===
os.makedirs("../figures", exist_ok=True)

# Accuracy
plt.figure(figsize=(10, 5))
plt.bar(results.keys(), [r["accuracy"] for r in results.values()], color="skyblue")
plt.xticks(rotation=15)
plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison")
plt.tight_layout()
plt.savefig("../figures/model_accuracy.png")

# F1-score
plt.figure(figsize=(10, 5))
plt.bar(results.keys(), [r["f1"] for r in results.values()], color="salmon")
plt.xticks(rotation=15)
plt.ylabel("F1-score")
plt.title("Model F1-score Comparison")
plt.tight_layout()
plt.savefig("../figures/model_f1score.png")

print("📊 模型比較圖已儲存 figures/model_accuracy.png / model_f1score.png")

# === 9. 繪製混淆矩陣 ===
plt.figure(figsize=(5, 4))
cm = confusion_matrix(y_test, results[best_tag]["y_pred"])
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title(f"Confusion Matrix ({best_tag})")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("../figures/confusion_matrix.png")
print("📊 混淆矩陣已儲存 figures/confusion_matrix.png")
