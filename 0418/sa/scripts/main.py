# analysis_main.py

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    mean_squared_error, r2_score
)
import joblib  # 用於儲存模型

from scripts.data_preprocessing import simulate_customer_data

# 設定視覺風格
sns.set(style="whitegrid")

# 設定資料夾路徑
FIGURE_DIR = "figures"
MODEL_DIR = "models"
RESULT_DIR = "results"
os.makedirs(FIGURE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# 1. 模擬顧客資料
print("\n=== 資料模擬階段 ===")
df = simulate_customer_data()
print(df.head())

# ----------------------
# 分群任務
# ----------------------
print("\n=== 分群任務 (Clustering) ===")
clustering_features = ['Age', 'Income', 'PurchaseFreq']
scaler = StandardScaler()
X_cluster = scaler.fit_transform(df[clustering_features])

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_cluster)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_cluster)
df['Cluster'] = cluster_labels

plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=cluster_labels, palette='Set2', s=60)
plt.title('KMeans 分群視覺化 (經 PCA 降維)', fontsize=14)
plt.xlabel('主成分 1')
plt.ylabel('主成分 2')
plt.legend(title='群編號')
plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR, 'clustering_pca.png'))
plt.show()

print("\n各群組在原始特徵的平均值：")
print(df.groupby('Cluster')[clustering_features].mean())

# ----------------------
# 分類任務
# ----------------------
print("\n=== 分類任務 (Classification) ===")
X_cls = df[['Age', 'Income', 'PurchaseFreq', 'Membership']]
y_cls = df['Segment']

X_train, X_test, y_train, y_test = train_test_split(X_cls, y_cls, test_size=0.3, random_state=42, stratify=y_cls)
clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("\n--- 分類報告 ---")
report_dict = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()
report_df.to_csv(os.path.join(RESULT_DIR, 'classification_report.csv'))
print(report_df)

cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_).plot(cmap='Blues', ax=ax)
plt.title('混淆矩陣：Segment 分類')
plt.grid(False)
plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR, 'classification_confusion_matrix.png'))
plt.show()

print("\n特徵重要性：")
importances = pd.Series(clf.feature_importances_, index=X_cls.columns).sort_values(ascending=False)
print(importances)
plt.figure(figsize=(6, 4))
importances.plot(kind='bar', title='分類任務：特徵重要性')
plt.ylabel('重要程度')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR, 'classification_feature_importance.png'))
plt.show()

# 儲存模型
joblib.dump(clf, os.path.join(MODEL_DIR, 'classifier.pkl'))

# ----------------------
# 回歸任務
# ----------------------
print("\n=== 回歸任務 (Regression) ===")
X_reg = df[['Age', 'Income', 'PurchaseFreq', 'Membership']]
y_reg = df['PurchaseAmount']

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg, test_size=0.3, random_state=42)
reg = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
reg.fit(X_train_r, y_train_r)
y_pred_r = reg.predict(X_test_r)

rmse = mean_squared_error(y_test_r, y_pred_r, squared=False)
r2 = r2_score(y_test_r, y_pred_r)

print(f"\n--- 回歸模型評估 ---")
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.2f}")

# 輸出回歸評估指標為 CSV
reg_metrics = pd.DataFrame({
    'RMSE': [rmse],
    'R2': [r2]
})
reg_metrics.to_csv(os.path.join(RESULT_DIR, 'regression_metrics.csv'), index=False)

plt.figure(figsize=(6, 5))
plt.scatter(y_test_r, y_pred_r, alpha=0.6)
plt.plot([y_test_r.min(), y_test_r.max()], [y_test_r.min(), y_test_r.max()], 'r--', label='理想預測線')
plt.xlabel('實際購買金額')
plt.ylabel('預測購買金額')
plt.title('回歸任務：預測 vs 實際')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR, 'regression_prediction.png'))
plt.show()

print("\n回歸任務特徵重要性：")
reg_importances = pd.Series(reg.feature_importances_, index=X_reg.columns).sort_values(ascending=False)
print(reg_importances)
plt.figure(figsize=(6, 4))
reg_importances.plot(kind='bar', title='回歸任務：特徵重要性')
plt.ylabel('重要程度')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR, 'regression_feature_importance.png'))
plt.show()

# 儲存模型
joblib.dump(reg, os.path.join(MODEL_DIR, 'regressor.pkl'))

print("\n=== 所有任務完成 ===")
