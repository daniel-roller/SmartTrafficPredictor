# train_regression.py
# 回歸任務：預測 Titanic 乘客的票價（Fare），包含多種模型與參數組合比較

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os
from preprocessing import preprocess_titanic

# === 1. 載入前處理後的 Titanic 資料 ===
df = preprocess_titanic("../data/train.csv", is_train=True)
X = df[["Pclass", "Sex", "Age", "SibSp", "Parch", "Embarked"]]
y = df["Fare"]

# === 2. 分割資料集 ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === 3. 建立通用前處理器 ===
num_features = ["Pclass", "Age", "SibSp", "Parch"]
cat_features = ["Sex", "Embarked"]
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_features),
    ("cat", OneHotEncoder(drop="first"), cat_features)
])

# === 4. 定義多種回歸模型與參數組合 ===
models = {
    "LinearRegression": [
        ("default", LinearRegression())
    ],
    "RandomForestRegressor": [
        ("n50_d5", RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)),
        ("n150_d10", RandomForestRegressor(n_estimators=150, max_depth=10, random_state=42))
    ],
    "SVR": [
        ("C1", SVR(C=1, kernel='rbf')),
        ("C10", SVR(C=10, kernel='rbf'))
    ]
}

# === 5. 模型訓練與評估 ===
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
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        results[tag] = {
            "mse": mse,
            "r2": r2,
            "model": pipe
        }
        print(f"{tag} ➜ MSE = {mse:.2f}, R2 = {r2:.4f}")

# === 6. 找出最佳模型（以 MSE 為主） ===
best_tag = min(results, key=lambda k: results[k]["mse"])
best_model = results[best_tag]["model"]
print(f"\n✅ Best model: {best_tag}")

# === 7. 儲存最佳模型 ===
os.makedirs("../model", exist_ok=True)
joblib.dump(best_model, "../model/titanic_best_regressor.pkl")
print("✅ 模型已儲存為 model/titanic_best_regressor.pkl")

# === 8. 繪製比較圖（MSE） ===
os.makedirs("../figures", exist_ok=True)
plt.figure(figsize=(10, 5))
plt.bar(results.keys(), [r["mse"] for r in results.values()], color="skyblue")
plt.xticks(rotation=15)
plt.ylabel("Mean Squared Error")
plt.title("Regression Model Comparison (MSE)")
plt.tight_layout()
plt.savefig("../figures/regression_comparison.png")
print("📊 模型 MSE 圖已儲存 figures/regression_comparison.png")
