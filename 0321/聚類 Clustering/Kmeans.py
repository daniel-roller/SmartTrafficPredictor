import os
os.environ["OMP_NUM_THREADS"] = "1"  # 解決 KMeans 記憶體問題

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

plt.rcParams['font.family'] = 'Microsoft JhengHei'  # 解決中文顯示問題


# 模擬每小時車流量（24小時 × 7天 = 168筆）
hours = list(range(24)) * 7
traffic = [50 + 30*np.sin((h-7)/12*np.pi) + np.random.normal(0,5) for h in hours]
df = pd.DataFrame({"hour": hours, "traffic": traffic})

# 特徵標準化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[["hour", "traffic"]])

# 執行 K-Means（分3群）
kmeans = KMeans(n_clusters=3, random_state=0)
df["cluster"] = kmeans.fit_predict(X_scaled)

# 畫圖看分群結果
plt.scatter(df.index, df["traffic"], c=df["cluster"], cmap="viridis", s=30)
plt.title("K-Means 聚類：時段車流型態")
plt.xlabel("時間（小時）")
plt.ylabel("車流量")
plt.grid(True)
plt.show()
