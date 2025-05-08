from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd

# 1. 取得加州房價資料
data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# 2. 切分訓練集和測試集（8成訓練，2成測試）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. 建立線性回歸模型
model = LinearRegression()
model.fit(X_train, y_train)

# 4. 評估模型
score = model.score(X_test, y_test)
print(f"模型的準確度(R²): {score:.2f}")

# 5. 繪圖：預測 vs. 真實值
y_pred = model.predict(X_test)
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("True Price")
plt.ylabel("Predicted Price")
plt.title("California Housing Price Prediction")
plt.grid(True)
plt.show()
