from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = SVC(kernel='rbf', gamma='scale')  # 使用 RBF 核（高斯核）
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"RBF SVM Accuracy: {accuracy:.2f}")
"""
適用於非線性資料，常見選擇，能處理複雜邊界
RBF SVM（高斯/徑向基核）

"""