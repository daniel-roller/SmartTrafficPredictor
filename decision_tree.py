from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# 載入 Iris 資料集
iris = load_iris()
X = iris.data
y = iris.target

# 切分資料集為訓練集與測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 建立並訓練決策樹模型
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# 預測並計算準確率
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# 顯示結果
print(f"Accuracy: {accuracy:.2f}")
"""
load_iris()：載入 Iris 資料集，這是一個常用於分類問題的資料集。

train_test_split()：將資料集分為訓練集和測試集，30% 用來測試，70% 用來訓練。

DecisionTreeClassifier()：建立一個決策樹分類模型。

fit()：用訓練集資料來訓練模型。

predict()：對測試集資料進行預測。

accuracy_score()：計算並顯示模型在測試集上的準確率。

優點：
容易理解：決策樹像樹一樣，每個分支可以直觀解釋。

能處理複雜資料：可以處理非線性關係的資料。

不需要資料標準化：不會受資料尺度影響。

能處理缺失值：對缺失資料有容錯性。

缺點：
容易過擬合：如果樹太深，模型會對訓練資料過於精確，表現不好。

對異常值敏感：極端數據會影響結果。

決策邊界不平滑：用直線或垂直分割，無法處理曲線型資料。

大樹難以解釋：樹太深時，解釋變得複雜。

"""