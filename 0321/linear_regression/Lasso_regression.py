# Lasso 回歸（L1 正則化）
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error

# 載入資料集
boston = load_boston()
X = boston.data
y = boston.target

# 切分資料集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 訓練 Lasso 回歸模型（L1 正則化）
model = Lasso(alpha=0.1)  # alpha 控制正則化強度
model.fit(X_train, y_train)

# 預測並計算均方誤差（MSE）
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")
