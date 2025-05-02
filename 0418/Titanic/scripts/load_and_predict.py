# load_and_predict.py
# 完全修正版：正確預測 train.csv 與 test.csv，產生 prediction 與 submission

import pandas as pd
import joblib
import os
from preprocessing import preprocess_titanic

# === 1. 載入最新模型 ===
model_path = "../model/titanic_best_model.pkl"
if not os.path.exists(model_path):
    raise FileNotFoundError("找不到模型檔案，請先執行 train_and_compare.py")

model = joblib.load(model_path)
print("✅ 已成功載入模型")


# === 2. 預測 train.csv 資料 ===
print("\n🔹 預測 train.csv 資料...")

df_train = preprocess_titanic("../data/train.csv", is_train=True)
X_train = df_train[["Pclass", "Fare", "SibSp", "Parch", "Sex", "Embarked"]]

# 預測 train.csv
y_pred_train = model.predict(X_train)

# 儲存 train 預測結果
os.makedirs("../output", exist_ok=True)
df_result = X_train.copy()
df_result["predicted_survived"] = y_pred_train
df_result.to_csv("../output/titanic_prediction_result.csv", index=False)
print("📅 預測結果已儲存 output/titanic_prediction_result.csv")


# === 3. 預測 test.csv 生成 submission.csv ===
print("\n🔹 預測 test.csv 產生 submission.csv...")

df_test, passenger_ids = preprocess_titanic("../data/test.csv", is_train=False)
X_test = df_test[["Pclass", "Fare", "SibSp", "Parch", "Sex", "Embarked"]]

# 預測 test.csv
y_submit_pred = model.predict(X_test)

# 儲存 submission.csv
submission = pd.DataFrame({
    "PassengerId": passenger_ids,
    "Survived": y_submit_pred
})
submission.to_csv("../output/submission.csv", index=False)
print("📅 submission.csv 已儲存 output/submission.csv，可上傳 Kaggle!")

print("\n🚀 全部預測結束！")
# 這段程式碼會載入訓練好的模型，然後預測 train.csv 和 test.csv 的資料，並將預測結果儲存到指定的 CSV 檔案中。
# 這樣的結構讓你可以輕鬆地將預測結果與原始資料進行比較，並且能夠生成 Kaggle 所需的 submission 格式。
# 這樣的設計也讓你可以輕鬆地將預測結果與原始資料進行比較，並且能夠生成 Kaggle 所需的 submission 格式。