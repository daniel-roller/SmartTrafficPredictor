from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. 載入鳶尾花數據集
data = load_iris()
X, y = data.data, data.target

# 2. 切分數據（80% 訓練，20% 測試）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. 建立隨機森林分類器
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# 4. 訓練模型
clf.fit(X_train, y_train)

# 5. 進行預測
y_pred = clf.predict(X_test)

# 6. 計算準確率
accuracy = accuracy_score(y_test, y_pred)
print(f"模型準確率: {accuracy:.2f}")