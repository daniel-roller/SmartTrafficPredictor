#Import Required Libraries

#匯入 pandas 套件，常用於讀取與處理表格型資料(ex : csv)
import pandas as pd
#匯入 NumPy，提供支援陣列、數值運算、統計等功能
import numpy as np
#匯入 os 模組，讓你能操作系統路徑、檔案、目錄等
import os
#匯入 matplotlib 的 pyplot 模組，通常用來畫圖或資料視覺化
import matplotlib.pyplot as plt
#匯入 seaborn，它是建立在 matplotlib 之上的統計視覺化套件，風格較美觀
import seaborn as sns
#關閉所有警告訊息
import warnings
warnings.filterwarnings('ignore')

#Load the dataset

#使用 pandas 讀取名為 Iris.csv 的 CSV 檔案，並存入 df（dataframe）變數中
df = pd.read_csv('Iris.csv')
#顯示資料表（dataframe）前 5 筆資料，方便快速瀏覽欄位與數據格式
df.head()

#移除資料表中不需要的 Id 欄位，因為這個欄位只是編號，不影響分析
df = df.drop(columns=["Id"])
#再次顯示前 5 筆資料（這時 Id 欄位已經移除了）
df.head()

#顯示資料表中 前 10 筆資料，可以看到更多花卉樣本的詳細數據
df.head(10)

"""
使用 pandas 的 describe() 方法，顯示數值型欄位的基本統計資料，包括：
count：非空值的數量
mean：平均值
std：標準差（資料離散程度）
min：最小值
25%、50%（中位數）、75%：分位數（描述資料分布）
max：最大值
通常用來快速了解資料的範圍與分布
"""
df.describe()

""""
使用 info() 方法來顯示資料表的整體結構資訊，包括：
每個欄位的名稱與順序
欄位的資料型態（例如 float64, int64, object 等）
非空值的筆數（可用來判斷是否有缺漏資料）
記憶體使用量
非常適合用來檢查是否有 缺失值 或 類別資料
"""
df.info()

#顯示 Species 欄位中每個類別（花卉品種）出現的次數
df['Species'].value_counts()

#Preprocessing the Dataset

#從 scikit-learn 的 preprocessing 模組中匯入 LabelEncoder，這個類別可以把「字串類別」轉換成「數值編號」
from sklearn.preprocessing import LabelEncoder
#建立一個 LabelEncoder 的實例（物件），命名為 le，準備對類別欄位進行轉換
le = LabelEncoder()
#對 Species 欄位的每個類別進行轉換，將字串類別變成整數編號，並把結果存回 df['Species']
df['Species'] = le.fit_transform(df['Species'])
#顯示轉換後的 Species 欄位內容，也就是每筆資料現在都變成數字 0、1、2 的形式
df['Species']

#顯示整個 DataFrame（df），此時 Species 欄位已經從字串變成數字，方便模型訓練
df

"""
檢查資料是否有缺失值
df.isnull()
對整個 DataFrame 執行「是否為缺失值」的判斷，會回傳一個布林值表格（True 代表該格是 null）
.sum()
對每個欄位將 True 數量加總（因為 True 被視為 1，False 為 0），得到每一欄中「缺失值的數量」
"""
df.isnull().sum()

#Exploratory Data Analysis (EDA)

# Plot histograms of each feature
df['SepalLengthCm'].hist()

df['SepalWidthCm'].hist()

df['PetalLengthCm'].hist()

df['PetalWidthCm'].hist()

#Plotting the histogram of all features toghether
df['SepalLengthCm'].hist()
df['SepalWidthCm'].hist()
df['PetalLengthCm'].hist()
df['PetalWidthCm'].hist()

# Plot scatterplots to visualize relationships between features
colors = ['red', 'orange', 'blue']
species = [0, 1, 2]

# Scatter plot for Sepal Length vs Sepal Width
for i in range(3):
    x = df[df['Species'] == species[i]]
    plt.scatter(x['SepalLengthCm'], x['SepalWidthCm'], c=colors[i], label=species[i])
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.legend()

# Scatter plot for Petal Length vs Petal Width 
for i in range(3):
    x = df[df['Species'] == species[i]]
    plt.scatter(x['PetalLengthCm'], x['PetalWidthCm'], c = colors[i], label=species[i])
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.legend()

# Scatter plot for Petal Length vs Sepal Length
for i in range(3):
    x = df[df['Species'] == species[i]]
    plt.scatter(x['SepalLengthCm'], x['PetalLengthCm'], c = colors[i], label=species[i])
plt.xlabel("Sepal Length")
plt.ylabel("Petal Length")
plt.legend()

# Scatter plot for Sepal Width vs Petal Width
for i in range(3):
    x = df[df['Species'] == species[i]]
    plt.scatter(x['SepalWidthCm'], x['PetalWidthCm'], c = colors[i], label=species[i])
plt.xlabel("Sepal Width")
plt.ylabel("Petal Width")
plt.legend()

#Correlation 

# Compute the correlation matrix 
df.corr()

# display the correlation matrix using a heatmap
corr = df.corr()
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(corr, annot=True, ax=ax, cmap='coolwarm')

#Model Training

# Split the dataset into training and testing sets
from sklearn.model_selection import train_test_split
X = df.drop(columns=['Species'])
Y = df['Species']
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.40)

# Logistic Regression Model
from sklearn.linear_model import LogisticRegression
model1 = LogisticRegression()
model1.fit(x_train, y_train)
print("Accuracy (Logistic Regression): ", model1.score(x_test, y_test) * 100)

# K-nearest Neighbours Model (KNN)
from sklearn.neighbors import KNeighborsClassifier
model2 = KNeighborsClassifier()
model2.fit(x_train, y_train)
print("Accuracy (KNN): ", model2.score(x_test, y_test) * 100)

# Decision Tree Model
from sklearn.tree import DecisionTreeClassifier
model3 = DecisionTreeClassifier()
model3.fit(x_train, y_train)
print("Accuracy (Decision Tree): ", model3.score(x_test, y_test) * 100)

#Confusion Matrix

from sklearn.metrics import confusion_matrix

y_pred1 = model1.predict(x_test)
y_pred2 = model2.predict(x_test)
y_pred3 = model3.predict(x_test)

conf_matrix1 = confusion_matrix(y_test, y_pred1)
conf_matrix2 = confusion_matrix(y_test, y_pred2)
conf_matrix3 = confusion_matrix(y_test, y_pred3)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix1, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(Y), yticklabels=np.unique(Y))
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix of Logistic Regression')
plt.show()

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix2, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(Y), yticklabels=np.unique(Y))
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix of KNN')
plt.show()

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix3, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(Y), yticklabels=np.unique(Y))
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix of Decision Tree')
plt.show()