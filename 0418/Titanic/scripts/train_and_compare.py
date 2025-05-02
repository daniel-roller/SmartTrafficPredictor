# train_and_compare.py
# 📊 用 Kaggle Titanic 資料，訓練、比較、畫圖、儲存最佳模型

import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

# === 1. 載入 Titanic 資料集 ===
from preprocessing import preprocess_titanic

# 處理 train.csv
df_train = preprocess_titanic("../data/train.csv", is_train=True)

# === 3. 特徵與目標 ===
X = df_train[["Pclass", "Fare", "SibSp", "Parch", "Sex", "Embarked"]]
y = df_train["Survived"]


cat_features = ["Sex", "Embarked"]
num_features = ["Pclass", "Fare", "SibSp", "Parch"]

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_features),
    ("cat", OneHotEncoder(drop="first"), cat_features)
])

# === 4. 資料切分 ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === 5. 定義模型（含合理超參數） ===
models = {
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
    "SVM (RBF Kernel)": SVC(kernel="rbf", C=1, gamma="scale", probability=True, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42)
}

# === 6. 訓練、預測、比較 ===
scores_acc = {}
scores_f1 = {}
best_model = None
best_score = 0
best_name = ""

for name, model in models.items():
    pipe = Pipeline([
        ("pre", preprocessor),
        ("model", model)
    ])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    scores_acc[name] = acc
    scores_f1[name] = f1

    print(f"🔹 {name} - Accuracy: {acc:.4f}, F1-score: {f1:.4f}")

    if acc > best_score:
        best_score = acc
        best_model = pipe
        best_name = name

# === 7. 儲存最佳模型 ===
joblib.dump(best_model, "../model/titanic_best_model.pkl")
print(f"\n✅ Best model: {best_name} 已儲存到 model/titanic_best_model.pkl")

# === 8. 畫圖（模型準確率、F1-score） ===
plt.figure(figsize=(8, 5))
plt.bar(scores_acc.keys(), scores_acc.values(), color="skyblue")
plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison")
plt.xticks(rotation=10)
plt.tight_layout()
plt.savefig("../figures/model_accuracy.png")
print("📊 準確率比較圖已儲存 figures/model_accuracy.png")

plt.figure(figsize=(8, 5))
plt.bar(scores_f1.keys(), scores_f1.values(), color="lightgreen")
plt.ylabel("F1-Score")
plt.title("Model F1-Score Comparison")
plt.xticks(rotation=10)
plt.tight_layout()
plt.savefig("../figures/model_f1score.png")
print("📊 F1-score 比較圖已儲存 figures/model_f1score.png")

# === 9. 混淆矩陣（最佳模型） ===
y_best_pred = best_model.predict(X_test)
cm = confusion_matrix(y_test, y_best_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Died", "Survived"])
disp.plot(cmap="Blues")
plt.title(f"Confusion Matrix - {best_name}")
plt.tight_layout()
plt.savefig("../figures/confusion_matrix.png")
print("📊 混淆矩陣圖已儲存 figures/confusion_matrix.png")

# === 10. 特徵重要性（如果有） ===
if hasattr(best_model.named_steps["model"], "feature_importances_"):
    feature_names = best_model.named_steps["pre"].get_feature_names_out()
    importances = best_model.named_steps["model"].feature_importances_

    plt.figure(figsize=(8, 6))
    plt.barh(feature_names, importances, color="salmon")
    plt.xlabel("Importance")
    plt.title(f"Feature Importances - {best_name}")
    plt.tight_layout()
    plt.savefig("../figures/feature_importance.png")
    print("📊 特徵重要性圖已儲存 figures/feature_importance.png")
else:
    print(f"⚠️ {best_name} 不支援特徵重要性（例如 SVM、Logistic）")

