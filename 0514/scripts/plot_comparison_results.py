import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
os.makedirs("figures", exist_ok=True)
# 讀取資料
df = pd.read_csv("comparison_results.csv")
df["Dataset"] = df["Dataset"].str.strip()
df["Model"] = df["Model"].str.strip()

sns.set(style="whitegrid")

# === 畫 F1 Score 圖並儲存 ===
plt.figure(figsize=(14, 6))
sns.barplot(data=df, x="Model", y="F1 Score", hue="Dataset")
plt.title("三個資料集各模型的 F1 Score 比較）", fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("figures/f1_score_comparison.png")
plt.show()

# === 畫 Accuracy 圖並儲存 ===
plt.figure(figsize=(14, 6))
sns.barplot(data=df, x="Model", y="Accuracy", hue="Dataset")
plt.title("三個資料集各模型的 Accuracy 比較", fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("figures/accuracy_comparison.png")
plt.show()
