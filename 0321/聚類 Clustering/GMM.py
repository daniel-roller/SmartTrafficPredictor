import os
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

plt.rcParams['font.family'] = 'Microsoft JhengHei'

# 模擬資料
hours = list(range(24)) * 7
traffic = [50 + 30*np.sin((h-7)/12*np.pi) + np.random.normal(0, 5) for h in hours]
df = pd.DataFrame({"hour": hours, "traffic": traffic})

# 標準化
X = StandardScaler().fit_transform(df[["hour", "traffic"]])

# GMM 分群
gmm = GaussianMixture(n_components=3, random_state=0)
df["cluster"] = gmm.fit_predict(X)

# 畫圖
plt.figure(figsize=(12, 5))
plt.scatter(df.index, df["traffic"], c=df["cluster"], cmap="plasma", s=30)
plt.title("車流量聚類：GMM 分群")
plt.xlabel("時間（小時）")
plt.ylabel("車流量")
plt.grid(True)
plt.show()
