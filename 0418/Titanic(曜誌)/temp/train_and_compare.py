# train_and_compare.py
# 🛠️ 使用 Titanic 資料，訓練多個模型進行分類，並比較誰表現最好，再儲存最佳模型

# === 套件匯入 ===
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

# === 1. 載入前處理後的 Titanic 訓練資料 ===
from preprocessing import preprocess_titanic
df_train = preprocess_titanic("../data/train.csv", is_train=True)

# === 2. 選擇輸入欄位（X）與預測目標欄位（y） ===
X = df_train[["Pclass", "Fare", "SibSp", "Parch", "Sex", "Embarked"]]  # 特徵欄位
y = df_train["Survived"]  # 預測是否生還（0 or 1）

# === 3. 定義哪些欄位要標準化、哪些要做 OneHot 編碼 ===
cat_features = ["Sex", "Embarked"]
num_features = ["Pclass", "Fare", "SibSp", "Parch"]

# ColumnTransformer：數值欄位做標準化，類別欄位做 OneHot 編碼
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_features),
    ("cat", OneHotEncoder(drop="first"), cat_features)
])

# === 4. 分割資料為訓練集與測試集（80% 訓練, 20% 測試）===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === 5. 定義要比較的模型，包含常見的四種分類器 ===
models = {
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
    "SVM (RBF Kernel)": SVC(kernel="rbf", C=1, gamma="scale", probability=True, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42)
}

# === 6. 逐個模型進行訓練、預測與評估 ===
scores_acc = {}  # 儲存準確率
scores_f1 = {}   # 儲存 F1-score
best_model = None
best_score = 0
best_name = ""

for name, model in models.items():
    # 使用前處理 + 模型 組合成 Pipeline
    pipe = Pipeline([
        ("pre", preprocessor),
        ("model", model)
    ])
    pipe.fit(X_train, y_train)         # 模型訓練
    y_pred = pipe.predict(X_test)      # 模型預測

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    scores_acc[name] = acc
    scores_f1[name] = f1

    print(f"🔹 {name} - Accuracy: {acc:.4f}, F1-score: {f1:.4f}")

    # 找出準確率最高的模型
    if acc > best_score:
        best_score = acc
        best_model = pipe
        best_name = name

# === 7. 儲存最佳模型（之後可直接載入使用）===
joblib.dump(best_model, "../model/titanic_best_model.pkl")
print(f"\n✅ Best model: {best_name} 已儲存到 model/titanic_best_model.pkl")

# === 8. 畫出模型準確率比較圖 ===
plt.figure(figsize=(8, 5))
plt.bar(scores_acc.keys(), scores_acc.values(), color="skyblue")
plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison")
plt.xticks(rotation=10)
plt.tight_layout()
plt.savefig("../figures/model_accuracy.png")
print("📊 準確率比較圖已儲存 figures/model_accuracy.png")

# === 9. 畫出模型 F1-score 比較圖 ===
plt.figure(figsize=(8, 5))
plt.bar(scores_f1.keys(), scores_f1.values(), color="lightgreen")
plt.ylabel("F1-Score")
plt.title("Model F1-Score Comparison")
plt.xticks(rotation=10)
plt.tight_layout()
plt.savefig("../figures/model_f1score.png")
print("📊 F1-score 比較圖已儲存 figures/model_f1score.png")

# === 10. 畫出最佳模型的混淆矩陣（了解分類錯在哪）===
y_best_pred = best_model.predict(X_test)
cm = confusion_matrix(y_test, y_best_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Died", "Survived"])
disp.plot(cmap="Blues")
plt.title(f"Confusion Matrix - {best_name}")
plt.tight_layout()
plt.savefig("../figures/confusion_matrix.png")
print("📊 混淆矩陣圖已儲存 figures/confusion_matrix.png")

# === 11. 畫出特徵重要性（僅支援部分模型，如 Decision Tree） ===
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
