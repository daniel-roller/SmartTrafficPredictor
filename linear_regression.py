from sklearn.datasets import fetch_california_housing  # 載入加州房價資料集（內建範例資料）
from sklearn.model_selection import train_test_split    # 載入資料切分工具（訓練 / 測試）
from sklearn.linear_model import LinearRegression       # 載入線性迴歸模型
from sklearn.metrics import mean_squared_error, r2_score  # 載入模型評估指標
import matplotlib.pyplot as plt                         # 載入繪圖套件

# 1. 載入資料
data = fetch_california_housing()  # 從 sklearn 自帶資料集中載入加州房價資料
X, y = data.data, data.target      # X 是特徵（輸入資料），y 是目標（房價，單位為萬美元）

# 2. 將資料切成訓練集與測試集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,       # 測試資料占整體的 20%
    random_state=20      # 設定隨機種子，確保每次切資料的結果一樣（方便除錯與重現結果）
)

# 3. 建立線性回歸模型
model = LinearRegression(
    fit_intercept=True,  # 是否要自動學習截距項（預設為 True，代表 y = wx + b 中的 b 要一起學）
    copy_X=True,         # 是否要複製一份 X（預設 True，會保留原始資料）
    n_jobs=None          # 設定 CPU 核心數。None 表示使用 1 核心；-1 則會使用所有核心
)

# 4. 使用訓練資料訓練模型
model.fit(X_train, y_train)  # 執行訓練，讓模型學會各個特徵對預測結果的影響（係數）

# 5. 用訓練好的模型對測試資料進行預測
y_pred = model.predict(X_test)  # 根據 X_test 的輸入，預測對應的房價結果

# 6. 評估模型的預測準確度
mse = mean_squared_error(y_test, y_pred)  # 平均平方誤差（MSE）：預測值與實際值的差異平方平均，越小越好
r2 = r2_score(y_test, y_pred)             # 決定係數 R²：表示模型對資料的解釋能力，越接近 1 越好

# 7. 印出評估指標
print(f"Mean Squared Error (MSE): {mse:.2f}")  # 顯示誤差大小
print(f"R-squared (R²): {r2:.2f}")             # 顯示模型準確度

# 8. 畫圖：實際值 vs 預測值的散點圖
plt.figure(figsize=(8, 6))  # 設定圖表大小
plt.scatter(y_test, y_pred, alpha=0.5, color='blue')  # 每個點代表一筆資料的「實際值 vs 預測值」
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # 畫出理想狀況下的對角線（y = x）
plt.xlabel("Actual Price")     # X 軸標籤：實際房價
plt.ylabel("Predicted Price")  # Y 軸標籤：預測房價
plt.title("Linear Regression: Actual vs Predicted")  # 圖表標題
plt.grid(True)     # 顯示網格線
plt.tight_layout() # 自動調整圖表間距
plt.show()         # 顯示圖表

"""
補充說明：
- 加州房價資料集中，target（房價）是以「萬美元」為單位，且最大值被限制在約 5 左右。
- 所以預測時，看到很多資料集中在 5 附近是正常現象。
"""
