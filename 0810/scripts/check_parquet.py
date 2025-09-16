# check parquet
import pandas as pd
import os

# 路徑
parquet_path = os.path.join("..", "cleaned", "20240101.parquet")

# 讀 parquet
df = pd.read_parquet(parquet_path)

# 看前 10 筆
print(df.head(1000))

# 看資料大小
print("資料筆數：", len(df))
print("欄位：", df.columns.tolist())
