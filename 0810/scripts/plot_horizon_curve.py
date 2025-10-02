# plot_horizon_curve.py
import pandas as pd
import matplotlib.pyplot as plt
import re

summary_file = "../results/horizon_summary.txt"

# 解析 horizon_summary.txt
data = []
with open(summary_file, "r", encoding="utf-8") as f:
    for line in f:
        match = re.match(r"HORIZON=(\d+): MAE=([\d.]+), MAPE=([\d.]+)%", line.strip())
        if match:
            horizon = int(match.group(1))
            mae = float(match.group(2))
            mape = float(match.group(3))
            data.append({"HORIZON": horizon, "MAE": mae, "MAPE": mape})

df = pd.DataFrame(data).sort_values("HORIZON")

# 畫圖
plt.figure(figsize=(8,6))
plt.plot(df["HORIZON"], df["MAE"], marker="o", label="MAE")
plt.plot(df["HORIZON"], df["MAPE"], marker="s", label="MAPE")
plt.xlabel("HORIZON (15-min steps)")
plt.ylabel("Error")
plt.title("HORIZON vs Prediction Error")
plt.legend()
plt.grid(True)
plt.savefig("../results/horizon_curve.png")
plt.show()
