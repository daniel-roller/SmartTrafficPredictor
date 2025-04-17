# Ridge Regression（L2 正則化）

from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import Ridge # 加上了 L2 正則化，有助於避免模型過擬合。
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_squared_error # 用於計算均方誤差（MSE)

data = fetch_california_housing()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = Ridge(alpha=1.0)
"""
alpha=1.0 是 正則化強度：

數字越大，懲罰越強，模型會更簡單。

數字越小，懲罰越弱，模型趨近一般線性回歸。
"""
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Ridge MSE: {mse:.2f}")
