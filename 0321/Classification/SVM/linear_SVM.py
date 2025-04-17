from sklearn.datasets import load_iris  # 載入鳶尾花資料集
from sklearn.model_selection import train_test_split  # 資料切分工具
from sklearn.svm import SVC  # SVM 分類器
from sklearn.metrics import accuracy_score  # 準確率評估

iris = load_iris()  # 載入資料
X, y = iris.data, iris.target  # 特徵與目標

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # 分割資料

clf = SVC(kernel='linear')  # 使用線性核的 SVM
clf.fit(X_train, y_train)  # 訓練模型

y_pred = clf.predict(X_test)  # 預測測試資料
accuracy = accuracy_score(y_test, y_pred)  # 計算準確度
print(f"Linear SVM Accuracy: {accuracy:.2f}")  # 印出結果

"""
這段程式碼使用線性 SVM 分類器來預測鳶尾花資料集的類別，並計算準確度。

適用於線性可分資料，速度快，解釋性強
"""