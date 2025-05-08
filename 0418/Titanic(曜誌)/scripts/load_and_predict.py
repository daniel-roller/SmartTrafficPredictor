# load_and_predict.py（多模型版本）
# 一次載入所有模型（分類、回歸、分群）並針對 test 資料預測，結果儲存在 results 資料夾中

import os
import joblib
import pandas as pd
from preprocessing import preprocess_titanic

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
    }
}

# === 載入測試資料（並取得 PassengerId） ===
def load_test_data():
    X, ids = preprocess_titanic("../data/test.csv", is_train=False)
    return X, ids


# === 儲存預測結果 ===
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
    X_test, ids = load_test_data()

    for task, info in model_info.items():
        path = info["path"]
        output_csv = info["csv"]
        label_name = ""

        if not os.path.exists(path):
            print(f"⚠️ 模型不存在：{path}，略過 {task}")
            continue

        model = joblib.load(path)
        # 安全顯示模型名稱
        # 安全顯示模型名稱，不管是不是 pipeline
        try:
            model_type = type(model.named_steps["model"]).__name__
        except AttributeError:
            model_type = type(model).__name__
        except KeyError:
            model_type = type(model).__name__

        print(f"🔍 使用模型（{task}）：{model_type}")

        pred = model.predict(X_test)

        # 根據任務類型命名欄位
        if task == "classifier" or task == "tuned_classifier":
            label_name = "Survived"
        elif task == "regressor":
            label_name = "Fare_Predicted"
        elif task == "cluster":
            label_name = "Cluster"

        save_prediction(ids, pred, output_csv, label_name)

if __name__ == "__main__":
    main()
