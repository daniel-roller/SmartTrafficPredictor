# tune_heart_classification.py
# 調整 Heart Disease 分類模型參數 + SMOTE + SelectKBest + LightGBM + CatBoost

import shap
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix
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

# 建立 ColumnTransformer 後（你的 preprocessor）
preprocessor.fit(X_train)

# 取得 onehot 特徵名
ohe = preprocessor.named_transformers_["cat"]
ohe_feature_names = ohe.get_feature_names_out(cat_features)

# 組合全部特徵名
feature_names = num_features + list(ohe_feature_names)

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
            "select__k": [7, 10],
            "model__iterations": [50, 100, 200],
            "model__depth": [4, 6, 8],
            "model__learning_rate": [0.05, 0.1, 0.2]
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
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    results[name] = {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "model": grid.best_estimator_
    }

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix ({name})")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"../figures/heart_confusion_matrix_{name}.png")

    print(f"✅ {name} 最佳參數：{grid.best_params_}")
    print(f"➡️ 測試集 F1-score = {f1:.4f}, Accuracy = {acc:.4f}\\n")

# === 5. 儲存 F1-score 最佳者 ===
best_name = max(best_models, key=lambda n: f1_score(y_test, best_models[n].predict(X_test)))
best_model = best_models[best_name]

os.makedirs("../model", exist_ok=True)
joblib.dump(best_model, "../model/heart_best_tuned.pkl")
print(f"💾 已儲存最佳調參模型：{best_name} 至 ../model/heart_best_tuned.pkl")

# === 6. 模型評估圖表與混淆矩陣 ===


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

# === 7. 額外產出：SHAP Summary Plot for 多模型 ===
print("📊 正在產出 SHAP Summary Plot...")

for name, model in best_models.items():
    try:
        # 跳過不支援的模型（例如 KNN）
        if name == "KNN":
            print(f"⚠️  {name} 不支援 SHAP，跳過。")
            continue

        # 取得特徵名稱
        selector = model.named_steps["select"]
        selected_indices = selector.get_support(indices=True)
        pre_X = preprocessor.transform(X_test)
        selected_X = selector.transform(pre_X)
        selected_feature_names = [feature_names[i] for i in selected_indices]

        # 建立 explainer 並畫圖
        explainer = shap.Explainer(model.named_steps["model"])
        shap_values = explainer(selected_X)

        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, selected_X, feature_names=selected_feature_names, show=False)
        plt.title(f"SHAP Summary Plot ({name})")
        plt.tight_layout()
        plt.savefig(f"../figures/heart_shap_summary_{name}.png")
        plt.close()
        print(f"✅ 已儲存 SHAP Summary Plot（{name}）")

    except Exception as e:
        print(f"❌ SHAP 畫圖失敗：{name} → {e}")

# 匯出模型整體比較表格

for m in results:
    if "precision" not in results[m]:
        print(f"⚠️ 缺少 precision：{m}")

df_metrics = pd.DataFrame({
    "Model": list(results.keys()),
    "Accuracy": [results[m]["accuracy"] for m in results],
    "F1-score": [results[m]["f1"] for m in results],
    "Precision": [results[m]["precision"] for m in results],
    "Recall": [results[m]["recall"] for m in results]
})
df_metrics.to_csv("../figures/heart_model_comparison.csv", index=False)
print("📄 模型指標總表已儲存為 heart_model_comparison.csv")
print("✅ 完成所有模型訓練與評估！")

# 可視化指標比較（條狀圖）
df_metrics.set_index("Model")[["Accuracy", "F1-score", "Precision", "Recall"]].plot(
    kind="bar", figsize=(10, 6), colormap="viridis"
)
plt.ylabel("Score")
plt.ylim(0.7, 1.0)
plt.title("Heart Disease Model Performance Comparison")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("../figures/heart_model_comparison_bar.png")
print("📊 模型比較條狀圖已儲存為 heart_model_comparison_bar.png")
