# train_clustering.py
# 分群任務：對 Titanic 資料進行無監督學習，使用多組群數比較並繪製圖表

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import joblib
import numpy as np
import os
from preprocessing import preprocess_titanic

# === 1. 載入前處理後的 Titanic 資料（不含目標欄位） ===
df = preprocess_titanic("../data/train.csv", is_train=True)
X = df[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]]

# === 2. 建立通用前處理器 ===
num_features = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
cat_features = ["Sex", "Embarked"]
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_features),
    ("cat", OneHotEncoder(drop="first"), cat_features)
])

# === 3. 設定欲測試的群數 K 值 ===
k_values = [2, 3, 4, 5]
results = {}

# === 4. 依每個 K 建立 KMeans 模型，訓練、評估、儲存結果 ===
for k in k_values:
    pipe = Pipeline([
        ("pre", preprocessor),
        ("cluster", KMeans(n_clusters=k, random_state=42, n_init=10))
    ])
    pipe.fit(X)
    labels = pipe.named_steps["cluster"].labels_
    X_transformed = pipe.named_steps["pre"].transform(X)
    score = silhouette_score(X_transformed, labels)
    results[k] = {
        "score": score,
        "labels": labels,
        "pipe": pipe,
        "X_transformed": X_transformed
    }
    print(f"K={k} ➜ Silhouette Score = {score:.4f}")

# === 5. 找出最佳群數（分數最高） ===
best_k = max(results, key=lambda k: results[k]["score"])
best_model = results[best_k]["pipe"]
best_labels = results[best_k]["labels"]
X_best = results[best_k]["X_transformed"]

# === 6. PCA 降維到 2D 用於視覺化最佳分群 ===
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_best)

# === 7. 圖表儲存資料夾 ===
os.makedirs("../figures", exist_ok=True)
os.makedirs("../model", exist_ok=True)

# === 8. 分群分布圖（最佳群數） ===
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=best_labels, palette="Set2")
plt.title(f"Clustering Result (K={best_k})")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.tight_layout()
plt.savefig("../figures/clustering_distribution.png")
print("📊 最佳分群圖已儲存：clustering_distribution.png")

# === 9. 分群分數比較圖（Silhouette） ===
plt.figure(figsize=(7, 4))
plt.plot(list(results.keys()), [results[k]["score"] for k in results], marker='o', color='purple')
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.title("KMeans Clustering Score Comparison")
plt.grid(True)
plt.tight_layout()
plt.savefig("../figures/clustering_score_compare.png")
print("📊 分群分數比較圖已儲存：clustering_score_compare.png")

# === 10. 儲存最佳模型 ===
joblib.dump(best_model, f"../model/titanic_best_cluster.pkl")
print(f"✅ 已儲存最佳分群模型（K={best_k}）於 titanic_best_cluster.pkl")
