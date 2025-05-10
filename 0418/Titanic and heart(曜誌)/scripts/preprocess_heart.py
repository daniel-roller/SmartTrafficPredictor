
# preprocess_heart.py
# 前處理 Heart Disease 資料

import pandas as pd

def preprocess_heart(filepath):
    """
    前處理 Heart Disease 預測資料
    入口：
    - filepath: heart.csv 的檔案路徑

    回傳：
    - X: 特徵
    - y: 標籤（HeartDisease）
    """
    df = pd.read_csv(filepath)

    # 填補缺失值（如有）
    df = df.ffill()

    # 分出 X, y
    X = df.drop(columns=["HeartDisease"])
    y = df["HeartDisease"]

    return X, y
