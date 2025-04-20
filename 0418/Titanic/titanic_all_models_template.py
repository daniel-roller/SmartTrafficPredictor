
# 📌 Titanic 資料：分類 + 回歸 + 分群 任務比較（含註解）
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

# 分類模型
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# 回歸模型
from sklearn.linear_model import LinearRegression, Ridge

# 分群模型
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture

# 評估工具
from sklearn.metrics import accuracy_score, mean_squared_error

# 載入資料
df = sns.load_dataset("titanic")

# ========== 前處理 ==========
# 移除缺失過多欄位與不相關欄位
df = df.drop(columns=["deck", "embark_town", "alive", "who", "adult_male", "class"])

# 補年齡與embarked缺失值
df["age"].fillna(df["age"].median(), inplace=True)
df["embarked"].fillna(df["embarked"].mode()[0], inplace=True)

# 移除剩餘缺失值
df = df.dropna()

# 分類用：預測 survived
target_cls = "survived"

# 回歸用：預測 age
target_reg = "age"

# 分群用特徵（沒有 y）
features_cluster = ["fare", "pclass", "sibsp", "parch"]

# ========== 特徵轉換（共用） ==========
cat_features = ["sex", "embarked"]
num_features = ["pclass", "fare", "sibsp", "parch"]

preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), num_features),
    ("cat", OneHotEncoder(drop="first"), cat_features)
])

# ---------------------------------------
# 1️⃣ 分類任務：預測 survived
# ---------------------------------------
X_cls = df[num_features + cat_features]
y_cls = df[target_cls]

X_train, X_test, y_train, y_test = train_test_split(X_cls, y_cls, test_size=0.2, random_state=42)

pipe_tree = Pipeline([
    ("pre", preprocessor),
    ("model", DecisionTreeClassifier())
])

pipe_svc = Pipeline([
    ("pre", preprocessor),
    ("model", SVC())
])

pipe_tree.fit(X_train, y_train)
pipe_svc.fit(X_train, y_train)

print("🔹 分類任務（預測 survived）")
print(f"Decision Tree Accuracy: {pipe_tree.score(X_test, y_test):.2f}")
print(f"SVM Accuracy: {pipe_svc.score(X_test, y_test):.2f}")
print()

# ---------------------------------------
# 2️⃣ 回歸任務：預測 age（練習用）
# ---------------------------------------
X_reg = df[num_features + cat_features]
y_reg = df[target_reg]

X_train, X_test, y_train, y_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

pipe_lr = Pipeline([
    ("pre", preprocessor),
    ("model", LinearRegression())
])

pipe_ridge = Pipeline([
    ("pre", preprocessor),
    ("model", Ridge())
])

pipe_lr.fit(X_train, y_train)
pipe_ridge.fit(X_train, y_train)

y_pred_lr = pipe_lr.predict(X_test)
y_pred_ridge = pipe_ridge.predict(X_test)

print("🔹 回歸任務（預測 age）")
print(f"Linear Regression MSE: {mean_squared_error(y_test, y_pred_lr):.2f}")
print(f"Ridge Regression MSE: {mean_squared_error(y_test, y_pred_ridge):.2f}")
print()

# ---------------------------------------
# 3️⃣ 分群任務：KMeans / DBSCAN / GMM
# ---------------------------------------
X_cluster = df[features_cluster]
X_cluster_scaled = StandardScaler().fit_transform(X_cluster)

kmeans = KMeans(n_clusters=3, random_state=0)
labels_kmeans = kmeans.fit_predict(X_cluster_scaled)

db = DBSCAN(eps=0.7, min_samples=5)
labels_db = db.fit_predict(X_cluster_scaled)

gmm = GaussianMixture(n_components=3, random_state=0)
labels_gmm = gmm.fit_predict(X_cluster_scaled)

# 畫圖
plt.figure(figsize=(15, 4))
plt.subplot(1, 3, 1)
plt.title("KMeans 分群")
plt.scatter(X_cluster_scaled[:, 0], X_cluster_scaled[:, 1], c=labels_kmeans, cmap="Set1")

plt.subplot(1, 3, 2)
plt.title("DBSCAN 分群")
plt.scatter(X_cluster_scaled[:, 0], X_cluster_scaled[:, 1], c=labels_db, cmap="Set2")

plt.subplot(1, 3, 3)
plt.title("GMM 分群")
plt.scatter(X_cluster_scaled[:, 0], X_cluster_scaled[:, 1], c=labels_gmm, cmap="Set3")

plt.tight_layout()
plt.show()
