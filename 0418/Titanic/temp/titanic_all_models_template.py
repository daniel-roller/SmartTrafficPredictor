# 導入基本套件
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

# 限制 CPU 使用數量，避免 sklearn 警告
os.environ["LOKY_MAX_CPU_COUNT"] = "4"
os.environ["OMP_NUM_THREADS"] = "4"

# sklearn 處理流程需要的套件
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

# 評估指標
from sklearn.metrics import accuracy_score, mean_squared_error

# 載入 Titanic 資料集（來自 seaborn）
df = sns.load_dataset("titanic")

# ========== 資料前處理 ==========

# 移除缺失太多或不需要的欄位
df = df.drop(columns=["deck", "embark_town", "alive", "who", "adult_male", "class"])

# 補 age（用中位數）、embarked（用眾數）
df["age"] = df["age"].fillna(df["age"].median())
df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0])

# 移除剩下有缺值的資料
df = df.dropna()

# 設定三種任務的目標
target_cls = "survived"   # 分類用的目標
target_reg = "age"        # 回歸用的目標
features_cluster = ["fare", "pclass", "sibsp", "parch"]  # 分群只用這幾個欄位

# ========== 建立共用前處理流程 ==========

cat_features = ["sex", "embarked"]  # 類別欄位 → 做 OneHot 編碼
num_features = ["pclass", "fare", "sibsp", "parch"]  # 數值欄位 → 做標準化

# 整合數值 + 類別欄位的轉換流程
preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), num_features),
    ("cat", OneHotEncoder(drop="first"), cat_features)
])

# ---------------------------------------
# 1️⃣ 分類任務：預測是否生還（survived）
# ---------------------------------------

X_cls = df[num_features + cat_features]
y_cls = df[target_cls]

X_train, X_test, y_train, y_test = train_test_split(X_cls, y_cls, test_size=0.2, random_state=42)

# 建立 Decision Tree 流程（含前處理）
pipe_tree = Pipeline([
    ("pre", preprocessor),
    ("model", DecisionTreeClassifier())
])

# 建立 SVM 流程
pipe_svc = Pipeline([
    ("pre", preprocessor),
    ("model", SVC())
])

# 模型訓練
pipe_tree.fit(X_train, y_train)
pipe_svc.fit(X_train, y_train)

# 顯示分類準確率
print("🔹 分類任務（預測 survived）")
print(f"Decision Tree Accuracy: {pipe_tree.score(X_test, y_test):.2f}")
print(f"SVM Accuracy: {pipe_svc.score(X_test, y_test):.2f}")
print()

# ---------------------------------------
# 2️⃣ 回歸任務：預測乘客年齡（age）
# ---------------------------------------

X_reg = df[num_features + cat_features]
y_reg = df[target_reg]

X_train, X_test, y_train, y_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

# 線性回歸
pipe_lr = Pipeline([
    ("pre", preprocessor),
    ("model", LinearRegression())
])

# Ridge 回歸（加上 L2 正則化）
pipe_ridge = Pipeline([
    ("pre", preprocessor),
    ("model", Ridge())
])

# 模型訓練
pipe_lr.fit(X_train, y_train)
pipe_ridge.fit(X_train, y_train)

# 預測並計算誤差
y_pred_lr = pipe_lr.predict(X_test)
y_pred_ridge = pipe_ridge.predict(X_test)

# 顯示回歸誤差
print("🔹 回歸任務（預測 age）")
print(f"Linear Regression MSE: {mean_squared_error(y_test, y_pred_lr):.2f}")
print(f"Ridge Regression MSE: {mean_squared_error(y_test, y_pred_ridge):.2f}")
print()

# ---------------------------------------
# 3️⃣ 分群任務：KMeans / DBSCAN / GMM
# ---------------------------------------

from sklearn.decomposition import PCA

X_cluster = df[["fare", "pclass", "sibsp", "parch"]]

# 1️⃣ 先標準化數據（分群前很重要）
X_cluster_scaled = StandardScaler().fit_transform(X_cluster)

# 2️⃣ KMeans 分群
kmeans = KMeans(n_clusters=3, random_state=0)
labels_kmeans = kmeans.fit_predict(X_cluster_scaled)

# 3️⃣ DBSCAN 分群（可找異常）
db = DBSCAN(eps=0.7, min_samples=5)
labels_db = db.fit_predict(X_cluster_scaled)

# 4️⃣ GMM 分群（機率式）
gmm = GaussianMixture(n_components=3, random_state=0)
labels_gmm = gmm.fit_predict(X_cluster_scaled)

# 5️⃣ PCA 降維後視覺化三種分群方法
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_cluster_scaled)

plt.rcParams["font.family"] = "Microsoft JhengHei"
plt.rcParams["axes.unicode_minus"] = False
plt.figure(figsize=(15, 4))

plt.subplot(1, 3, 1)
plt.title("KMeans - PCA 可視化")
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels_kmeans, cmap='Set1')
plt.xlabel("主成分1")
plt.ylabel("主成分2")

plt.subplot(1, 3, 2)
plt.title("DBSCAN - PCA 可視化")
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels_db, cmap='Set2')
plt.xlabel("主成分1")

plt.subplot(1, 3, 3)
plt.title("GMM - PCA 可視化")
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels_gmm, cmap='Set3')
plt.xlabel("主成分1")

plt.tight_layout()
plt.show()

# 6️⃣ 群組統計特徵分析（以 KMeans 為例）
df_clustered = df.copy()
df_clustered["KMeans群"] = labels_kmeans

group_stats = df_clustered.groupby("KMeans群")[["fare", "pclass", "sibsp", "parch"]].mean()
print("📊 各群組特徵平均值：")
print(group_stats)
