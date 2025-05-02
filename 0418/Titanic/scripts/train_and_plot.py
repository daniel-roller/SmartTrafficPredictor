# 簡單版

import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# === 1. 載入 Kaggle 的 train.csv 資料 ===
df = pd.read_csv("../data/train.csv")

# === 2. 基本前處理 ===
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
df = df.drop(columns=["Cabin", "Ticket", "Name", "PassengerId"])
df = df.dropna()

# === 3. 特徵與目標欄位設定 ===
X = df[["Pclass", "Fare", "SibSp", "Parch", "Sex", "Embarked"]]
y = df["Survived"]

cat_features = ["Sex", "Embarked"]
num_features = ["Pclass", "Fare", "SibSp", "Parch"]

# 前處理器：標準化數值欄位、編碼類別欄位
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_features),
    ("cat", OneHotEncoder(drop="first"), cat_features)
])

# 資料切分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 定義模型們
models = {
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC()
}

scores = {}
best_model = None
best_score = 0
best_name = ""

# 訓練 + 評估
for name, model in models.items():
    pipe = Pipeline([
        ("pre", preprocessor),
        ("model", model)
    ])
    pipe.fit(X_train, y_train)
    acc = pipe.score(X_test, y_test)
    scores[name] = acc
    print(f"{name} Accuracy: {acc:.4f}")

    if acc > best_score:    
        best_score = acc
        best_model = pipe
        best_name = name

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ✅ 混淆矩陣
y_pred = best_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Died", "Survived"])
disp.plot(cmap="Blues")
plt.title(f"Confusion Matrix - {best_name}")
plt.tight_layout()
plt.savefig("../figures/confusion_matrix.png")
print("📊 混淆矩陣圖已儲存為 figures/confusion_matrix.png")

# ✅ 特徵重要性（只有決策樹類模型有 feature_importances_）
if hasattr(best_model.named_steps["model"], "feature_importances_"):
    # 取得所有欄位名稱
    feature_names = best_model.named_steps["pre"].get_feature_names_out()
    importances = best_model.named_steps["model"].feature_importances_

    # 畫圖
    plt.figure(figsize=(8, 5))
    plt.barh(feature_names, importances, color="lightseagreen")
    plt.xlabel("Importance")
    plt.title(f"Feature Importances - {best_name}")
    plt.tight_layout()
    plt.savefig("../figures/feature_importance.png")
    print("📊 特徵重要性圖已儲存為 figures/feature_importance.png")
else:
    print(f"⚠️ {best_name} 不支援特徵重要性（不是樹模型）")


# 儲存最佳模型
joblib.dump(best_model, "../model/titanic_best_model.pkl")
print(f"\n✅ Best model: {best_name} 已儲存到 ../model/titanic_best_model.pkl")

# ✅ 畫圖並儲存（升級版）
plt.figure(figsize=(8, 6))
bars = plt.bar(scores.keys(), scores.values(), color="skyblue")

# 找出最大值並標註不同顏色
best_model_name = max(scores, key=scores.get)
for bar, name in zip(bars, scores.keys()):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 0.01, f"{height:.2f}",
             ha='center', fontsize=12)
    if name == best_model_name:
        bar.set_color("deepskyblue")  # 最佳模型顏色加強

plt.ylabel("Accuracy", fontsize=14)
plt.title("Model Accuracy Comparison", fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.ylim(0, 1.05)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# 儲存
plt.savefig("../figures/model_accuracy.png", dpi=150)
print("📊 高解析度模型比較圖已儲存為 figures/model_accuracy.png")

