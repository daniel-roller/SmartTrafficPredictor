# 📌 Titanic 模型訓練與儲存（train_and_save_best_model.py）
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import joblib

# 載入資料（Seaborn 版本）
df = sns.load_dataset("titanic")
df = df.drop(columns=["deck", "embark_town", "alive", "who", "adult_male", "class"])
df["age"] = df["age"].fillna(df["age"].median())
df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0])
df = df.dropna()

# 分類任務
target_cls = "survived"
cat_features = ["sex", "embarked"]
num_features = ["pclass", "fare", "sibsp", "parch"]
X = df[num_features + cat_features]
y = df[target_cls]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 前處理 + 模型 Pipeline
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_features),
    ("cat", OneHotEncoder(drop="first"), cat_features)
])

pipe_tree = Pipeline([("pre", preprocessor), ("model", DecisionTreeClassifier())])
pipe_svc = Pipeline([("pre", preprocessor), ("model", SVC())])

pipe_tree.fit(X_train, y_train)
pipe_svc.fit(X_train, y_train)

score_tree = pipe_tree.score(X_test, y_test)
score_svc = pipe_svc.score(X_test, y_test)

if score_svc > score_tree:
    best_model = pipe_svc
    best_name = "SVM"
else:
    best_model = pipe_tree
    best_name = "Decision Tree"

# 儲存模型
joblib.dump(best_model, "titanic_best_model.pkl")
print(f"✅ 最佳模型：{best_name}（已儲存 titanic_best_model.pkl）")

