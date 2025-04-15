from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.tree import export_text

# 載入 Iris 數據集
iris = load_iris()
X = iris.data[:, 2:4]  # 只使用花瓣長度和花瓣寬度
y = iris.target

# 切分資料集為訓練集與測試集 (25% 測試資料，75% 訓練資料)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# 建立決策樹模型
model = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)

# 訓練模型
model.fit(X_train, y_train)

# 預測測試集資料
y_pred = model.predict(X_test)

# 計算準確率（Accuracy）
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
"""
優點：簡單、計算快、解釋性強、能夠提供特徵權重、能夠處理多個特徵。

缺點：不能處理非線性關係、對異常值敏感。

適用場景：當資料中變數之間有線性關係，並且需要快速且易於解釋的模型。
"""