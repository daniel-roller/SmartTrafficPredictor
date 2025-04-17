# 用來預測加州地區的房價
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# 載入加州房價資料集
california_housing = fetch_california_housing()
X = california_housing.data  # 特徵資料
y = california_housing.target  # 目標變數（房價）

# 切分資料集為訓練集與測試集（80% 訓練，20% 測試）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)

# 訓練 Ridge 回歸模型（L2 正則化）
model = Ridge(alpha=1.0)  # alpha 參數控制正則化強度
model.fit(X_train, y_train)

# 預測並計算均方誤差（MSE）
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"均方誤差(MSE): {mse:.2f}")
# 均方誤差是計算每一個預測值與實際值之間差異的平方，再將所有這些平方差的平均數
#合理範圍：MSE 越小越好，理想情況是 0