
# 📌 載入 Titanic 模型並預測（加強版）
import pandas as pd
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split

# 載入模型
model = joblib.load("titanic_best_model.pkl")
print("✅ 已成功載入 titanic_best_model.pkl")

# 印出模型摘要（顯示內部 Pipeline 結構）
print("\n🔍 模型摘要：")
print(model)

# 準備資料（與訓練時一致）
df = sns.load_dataset("titanic")
df = df.drop(columns=["deck", "embark_town", "alive", "who", "adult_male", "class"])
df["age"] = df["age"].fillna(df["age"].median())
df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0])
df = df.dropna()

cat_features = ["sex", "embarked"]
num_features = ["pclass", "fare", "sibsp", "parch"]
X = df[num_features + cat_features]
y = df["survived"]

_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 模型預測
y_pred = model.predict(X_test)

# 顯示前 10 筆預測
print("\n🔍 模型預測前 10 筆：")
print(y_pred[:10])

# 存成 CSV
df_result = X_test.copy()
df_result["predicted_survived"] = y_pred
df_result.to_csv("titanic_prediction_result.csv", index=False)
print("\n📁 預測結果已存為 titanic_prediction_result.csv")

# 額外：手動輸入新資料（只示範格式）
sample = pd.DataFrame([{
    "pclass": 1,
    "fare": 100,
    "sibsp": 0,
    "parch": 0,
    "sex": "female",
    "embarked": "S"
}])

new_pred = model.predict(sample)
print("\n🧪 新資料預測結果（是否生存）:", new_pred[0])
