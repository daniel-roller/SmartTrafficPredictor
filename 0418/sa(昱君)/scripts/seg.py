import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 模擬資料生成
np.random.seed(42)
n_samples = 300
age = np.random.randint(18, 65, size=n_samples)
income = np.random.randint(20000, 100000, size=n_samples)
purchase_freq = np.random.randint(1, 20, size=n_samples)
membership = np.random.choice([0, 1, 2, 3], size=n_samples)  # 會員等級

# 分類目標：有無購買（0/1）
target = (income > 60000).astype(int)

# 回歸目標：消費分數（與收入、年齡等有關）
spend_score = 0.4 * income / 1000 + 0.6 * purchase_freq + np.random.normal(0, 5, n_samples)

df = pd.DataFrame({
    'Age': age,
    'Income': income,
    'PurchaseFreq': purchase_freq,
    'Membership': membership,
    'Target': target,
    'SpendScore': spend_score
})
# 特徵與目標
X_cls = df[['Age', 'Income', 'PurchaseFreq', 'Membership']]
y_cls = df['Target']

# 分割資料
X_train, X_test, y_train, y_test = train_test_split(X_cls, y_cls, test_size=0.3, random_state=42)

# 使用 Logistic Regression 模型
clf = LogisticRegression()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# 評估
acc = accuracy_score(y_test, y_pred)
print("分類準確率：", acc)

# 混淆矩陣
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("分類任務：混淆矩陣")
plt.show()
X_reg = df[['Age', 'Income', 'PurchaseFreq', 'Membership']]
y_reg = df['SpendScore']

X_train, X_test, y_train, y_test = train_test_split(X_reg, y_reg, test_size=0.3, random_state=42)

reg = LinearRegression()
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print("回歸 MSE：", mse)

# 視覺化：預測 vs 實際
plt.scatter(y_test, y_pred)
plt.xlabel("實際 SpendScore")
plt.ylabel("預測 SpendScore")
plt.title("回歸任務：預測值 vs 實際值")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')
plt.show()
X_cluster = df[['Income', 'SpendScore']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

# 使用 KMeans 分群（測試 3 群）
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# 顯示群聚結果
sns.scatterplot(data=df, x='Income', y='SpendScore', hue='Cluster', palette='viridis')
plt.title("分群任務：Income vs SpendScore")
plt.show()
