import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Microsoft JhengHei'
plt.rcParams['axes.unicode_minus'] = False
os.makedirs("figures", exist_ok=True)

# 讀取資料
df = pd.read_csv("../results/comparison_results.csv")
df["Dataset"] = df["Dataset"].str.strip()
df["Model"] = df["Model"].str.strip()

sns.set(style="whitegrid")

# === Plot F1 Score ===
plt.figure(figsize=(14, 6))
sns.barplot(data=df, x="Model", y="F1 Score", hue="Dataset")
plt.title("F1 Score Comparison Across Models and Datasets", fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("figures/f1_score_comparison.png")
plt.show()

# === Plot Accuracy ===
plt.figure(figsize=(14, 6))
sns.barplot(data=df, x="Model", y="Accuracy", hue="Dataset")
plt.title("Accuracy Comparison Across Models and Datasets", fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("figures/accuracy_comparison.png")
plt.show()
