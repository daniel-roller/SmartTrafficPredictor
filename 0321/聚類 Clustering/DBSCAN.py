import os
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

plt.rcParams['font.family'] = 'Microsoft JhengHei'

# 模擬一週車流量資料
hours = list(range(24)) * 7
traffic = [50 + 30*np.sin((h-7)/12*np.pi) + np.random.normal(0, 5) for h in hours]
df = pd.DataFrame({"hour": hours, "traffic": traffic})

# 標準化
X = StandardScaler().fit_transform(df[["hour", "traffic"]])

# DBSCAN 分群
db = DBSCAN(eps=0.6, min_samples=5)
df["cluster"] = db.fit_predict(X)

# 畫圖
plt.figure(figsize=(12, 5))
plt.scatter(df.index, df["traffic"], c=df["cluster"], cmap="rainbow", s=30)
plt.title("車流量聚類：DBSCAN 分群（-1 表示離群點）")
plt.xlabel("時間（小時）")
plt.ylabel("車流量")
plt.grid(True)
plt.show()
