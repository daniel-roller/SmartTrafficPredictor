# preprocessing.py
# 💡 專門負責 Titanic 資料後處理模組

import pandas as pd

def preprocess_titanic(filepath, is_train=True):
    """
    📊 預處理 Titanic 資料

    入口：
    - filepath: 資料檔案路徑 (train.csv or test.csv)
    - is_train: True = 資料包含 Survived，False = 測試資料

    輸出：
    - 前處理後的 df，如果是測試資料，就同時返回 PassengerId
    """
    df = pd.read_csv(filepath)

    # 補值 Age、Fare、Embarked
    df["Age"] = df["Age"].fillna(df["Age"].median())
    if "Fare" in df.columns:
        df["Fare"] = df["Fare"].fillna(df["Fare"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

    # 如果是 test.csv，保留 PassengerId（後續用來做 submission.csv）
    passenger_ids = None
    if not is_train and "PassengerId" in df.columns:
        passenger_ids = df["PassengerId"]

    # 刪掉不用的欄位
    drop_cols = ["Cabin", "Ticket", "Name", "PassengerId"]
    df = df.drop(columns=[col for col in drop_cols if col in df.columns])

    # 保障沒有欠值
    df = df.dropna()

    if is_train:
        return df
    else:
        return df, passenger_ids
