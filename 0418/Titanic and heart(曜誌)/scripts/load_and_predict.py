# load_and_predict.py（分類 + 回歸 + 分群 + 心臟病任務）
# 載入所有模型並針對 test 資料預測，輸出結果到 results 資料夾中

import os
import joblib
import pandas as pd
from preprocessing import preprocess_titanic
from preprocess_heart import preprocess_heart

# === 模型與預測檔案對應表 ===
model_info = {
    "classifier": {
        "path": "../model/titanic_best_classification.pkl",
        "csv": "results/prediction_classification.csv"
    },
    "regressor": {
        "path": "../model/titanic_best_regressor.pkl",
        "csv": "results/prediction_regression.csv"
    },
    "cluster": {
        "path": "../model/titanic_best_cluster.pkl",
        "csv": "results/prediction_clustering.csv"
    },
    "tuned_classifier": {
        "path": "../model/titanic_best_tuned.pkl",
        "csv": "results/prediction_tuned_classification.csv"
    },
    "heart_tuned": {
        "path": "../model/heart_best_tuned.pkl",
        "csv": "results/prediction_heart_tuned.csv"
    }
}

# === 載入 Titanic 測試資料（含 PassengerId） ===
def load_test_data():
    X, ids = preprocess_titanic("../data/test.csv", is_train=False)
    return X, ids

# === 載入 Heart 測試資料（不含 PassengerId） ===
def load_heart_test_data():
    X, _ = preprocess_heart("../data/heart.csv")
    return X

# === 儲存 Titanic 的預測結果 ===
def save_prediction(ids, preds, filename, label_name):
    os.makedirs("results", exist_ok=True)
    df = pd.DataFrame({
        "PassengerId": ids,
        label_name: preds
    })
    df.to_csv(filename, index=False)
    print(f"✅ 已輸出預測結果至 {filename}")

# === 主流程 ===
def main():
    for task, info in model_info.items():
        path = info["path"]
        output_csv = info["csv"]

        if not os.path.exists(path):
            print(f"⚠️ 模型不存在：{path}，略過 {task}")
            continue

        model = joblib.load(path)

        # 決定使用哪一種測試資料與預測欄位名稱
        if "heart" in task:
            X_test = load_heart_test_data()
            label_name = "HeartDisease"
            pred = model.predict(X_test)
            os.makedirs("results", exist_ok=True)
            pd.DataFrame({label_name: pred}).to_csv(output_csv, index=False)
            print(f"✅ 已輸出預測結果至 {output_csv}")
        else:
            X_test, ids = load_test_data()
            label_name = ""
            if task in ["classifier", "tuned_classifier"]:
                label_name = "Survived"
            elif task == "regressor":
                label_name = "Fare_Predicted"
            elif task == "cluster":
                label_name = "Cluster"
            pred = model.predict(X_test)
            save_prediction(ids, pred, output_csv, label_name)

if __name__ == "__main__":
    main()
