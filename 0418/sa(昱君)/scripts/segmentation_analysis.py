import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import make_classification, make_regression
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.metrics import (
    classification_report, confusion_matrix,
    mean_squared_error, r2_score
)

# 設定視覺風格
sns.set(style="whitegrid")

# ---------- 1. 資料模擬 ----------
X_class, y_class = make_classification(
    n_samples=1000,
    n_features=4,
    n_informative=3,
    n_redundant=0,
    n_classes=3,
    n_clusters_per_class=1,
    random_state=42
)

X_reg, y_reg = make_regression(n_samples=1000, n_features=4, noise=10, random_state=42)

# 建立 DataFrame
df = pd.DataFrame(X_class, columns=['Age', 'Income', 'PurchaseFreq', 'Membership'])
df['Segment'] = y_class
df['PurchaseAmount'] = y_reg
df['Gender'] = np.random.choice(['Male', 'Female'], size=1000)

# ---------- 2. 分群分析 ----------
X_cluster = df[['Age', 'Income', 'PurchaseFreq']]
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_cluster)

# PCA 降維並繪圖
pca = PCA(n_components=2)
components = pca.fit_transform(X_cluster)
df['PC1'] = components[:, 0]
df['PC2'] = components[:, 1]

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='PC1', y='PC2', hue='Cluster', palette='Set2', s=60, alpha=0.8)
plt.title('PCA 可視化：KMeans 分群 (k=3)', fontsize=14)
plt.xlabel('主成分 1')
plt.ylabel('主成分 2')
plt.legend(title='群編號')
plt.tight_layout()
plt.show()

# ---------- 3. 分類任務 ----------
X_cls = df[['Age', 'Income', 'PurchaseFreq', 'Membership']]
y_cls = df['Segment']

X_train, X_test, y_train, y_test = train_test_split(X_cls, y_cls, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("\n--- 分類報告 (Segment 分類) ---")
print(classification_report(y_test, y_pred))

# 混淆矩陣圖
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('分類任務：混淆矩陣')
plt.xlabel('預測')
plt.ylabel('實際')
plt.tight_layout()
plt.show()

# ---------- 4. 回歸任務 ----------
X_reg = df[['Age', 'Income', 'PurchaseFreq', 'Membership']]
y_reg = df['PurchaseAmount']

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

reg = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
reg.fit(X_train_r, y_train_r)
y_pred_r = reg.predict(X_test_r)

rmse = mean_squared_error(y_test_r, y_pred_r, squared=False)
r2 = r2_score(y_test_r, y_pred_r)

print("\n--- 回歸評估 (PurchaseAmount 預測) ---")
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.2f}")

# 實際 vs 預測圖
plt.figure(figsize=(6, 4))
sns.scatterplot(x=y_test_r, y=y_pred_r, alpha=0.6)
plt.plot([y_test_r.min(), y_test_r.max()], [y_test_r.min(), y_test_r.max()], 'r--')
plt.xlabel('實際值')
plt.ylabel('預測值')
plt.title('回歸任務：實際 vs 預測')
plt.tight_layout()
plt.show()
