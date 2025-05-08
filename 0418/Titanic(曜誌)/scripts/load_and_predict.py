# load_and_predict.py（模組化版本）
# 根據模型類型自動預測 Titanic 測試資料並輸出結果

import os
import joblib
import pandas as pd
from preprocessing import preprocess_titanic

# === 1. 載入模型 ===
def load_model(path):
    assert os.path.exists(path), f"❌ 模型不存在：{path}"
    model = joblib.load(path)
    print("✅ 載入模型：", model.named_steps["model"])
    return model

# === 2. 載入測試資料（經過前處理） ===
def load_test_data(path):
    df = preprocess_titanic(path, is_train=False)
    return df.drop(columns=["PassengerId"]), df["PassengerId"]

# === 3. 模型預測 ===
def predict(model, X):
    return model.predict(X)

# === 4. 儲存預測結果 ===
def save_prediction(passenger_ids, predictions, model_type):
    df = pd.DataFrame({
        "PassengerId": passenger_ids,
        "Prediction": predictions
    })
    os.makedirs("../output", exist_ok=True)
    df.to_csv("../output/titanic_prediction_result.csv", index=False)
    print("📄 已輸出 titanic_prediction_result.csv")

    if model_type in ["DecisionTreeClassifier", "RandomForestClassifier", "SVC", "LogisticRegression"]:
        df.rename(columns={"Prediction": "Survived"}, inplace=True)
        df.to_csv("../output/submission.csv", index=False)
        print("📄 已輸出 Kaggle submission.csv")
    else:
        print("ℹ️ 非分類模型，不輸出 submission.csv")

# === 5. 主流程 ===
def main():
    model_path = "../model/titanic_best_model.pkl"  # 修改此路徑以測試不同模型
    test_path = "../data/test.csv"

    model = load_model(model_path)
    X_test, ids = load_test_data(test_path)
    model_type = type(model.named_steps["model"]).__name__
    predictions = predict(model, X_test)
    save_prediction(ids, predictions, model_type)

if __name__ == "__main__":
    main()