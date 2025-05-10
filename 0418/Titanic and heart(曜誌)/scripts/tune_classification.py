# tune_classification.py
# 🔍 加強版：分類模型參數調整 + SMOTE + SelectKBest

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
from preprocessing import preprocess_titanic
import joblib
import os

# === 1. 載入前處理後的 Titanic 資料 ===
df = preprocess_titanic("../data/train.csv", is_train=True)
X = df.drop(columns=["Survived"])
y = df["Survived"]

# === 2. 分割訓練/測試集 ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === 3. 建立前處理器 ===
num_features = ["Pclass", "Age", "SibSp", "Parch", "Fare", "FamilySize", "FarePerPerson"]
cat_features = ["Sex", "Embarked", "Title"]
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_features),
    ("cat", OneHotEncoder(drop="first"), cat_features)
])

# === 4. 建立模型與參數搜尋空間 ===
model_configs = {
    "XGBoost": {
        "model": XGBClassifier(eval_metric='logloss'),
        "params": {
            "select__k": [5, 8, 10],
            "model__n_estimators": [50, 100],
            "model__max_depth": [3, 5],
            "model__learning_rate": [0.05, 0.1]
        }
    },
    "KNN": {
        "model": KNeighborsClassifier(),
        "params": {
            "select__k": [5, 8, 10],
            "model__n_neighbors": [3, 5, 7],
            "model__weights": ["uniform", "distance"]
        }
    }
}

# === 5. 對每個模型進行 Grid Search ===
best_models = {}
for name, cfg in model_configs.items():
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
    print(f"✅ {name} 最佳參數：{grid.best_params_}")
    print(f"➡️ 測試集 F1-score = {f1:.4f}, Accuracy = {acc:.4f}\n")

# === 6. 儲存最佳模型（F1 分數高者） ===
best_name = max(best_models, key=lambda n: f1_score(y_test, best_models[n].predict(X_test)))
best_model = best_models[best_name]

os.makedirs("../model", exist_ok=True)
joblib.dump(best_model, "../model/titanic_best_tuned.pkl")
print(f"💾 已儲存最佳調參模型：{best_name} 至 titanic_best_tuned.pkl")
