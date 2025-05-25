# plot_results.py
import pandas as pd # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
import os
import numpy as np # type: ignore
from sklearn.metrics import confusion_matrix # type: ignore
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Microsoft JhengHei'
plt.rcParams['axes.unicode_minus'] = False
def plot_comparison_charts(csv_path="../results/comparison_results.csv", output_dir="../results"):
    # 讀取比較結果
    df = pd.read_csv(csv_path)

    # 設定繪圖樣式
    sns.set(style="whitegrid")

    # === F1 Score 圖表 ===
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x="Model", y="F1 Score", hue="Dataset")
    plt.xticks(rotation=45)
    plt.title("模型在不同資料集上的 F1 Score")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "f1_score_comparison.png"))
    plt.close()

    # === Accuracy 圖表 ===
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x="Model", y="Accuracy", hue="Dataset")
    plt.xticks(rotation=45)
    plt.title("模型在不同資料集上的 Accuracy")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "accuracy_comparison.png"))
    plt.close()
    
    # === Heapmap ===
    pivot_f1 = df.pivot(index="Model", columns="Dataset", values="F1 Score")
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot_f1, annot=True, cmap="YlGnBu", fmt=".2f")
    plt.title("F1 Score 熱力圖")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "f1_score_heatmap.png"))
    plt.close()
    
    # === Line Plot - F1 Score by Dataset ===
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="Dataset", y="F1 Score", hue="Model", marker="o")
    plt.title("各模型在不同資料集的 F1 Score 變化")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "f1_score_lineplot.png"))
    plt.close()

    # === Radar Chart for One Dataset (e.g. 第一個出現的 Dataset) ===
    dataset_name = df["Dataset"].unique()[0]
    subset = df[df["Dataset"] == dataset_name].sort_values("Model")
    labels = subset["Model"]
    stats = subset[["F1 Score", "Accuracy"]].values.T

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    stats = np.concatenate((stats, stats[:, [0]]), axis=1)
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.plot(angles, stats[0], label="F1 Score", linewidth=2)
    ax.fill(angles, stats[0], alpha=0.25)
    ax.plot(angles, stats[1], label="Accuracy", linewidth=2)
    ax.fill(angles, stats[1], alpha=0.25)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_title(f"{dataset_name} - Radar Chart (F1 & Accuracy)", y=1.08)
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "radar_chart.png"))
    plt.close()

    plot_confusion_matrices()
    print("📈 圖表已儲存至 results/ 資料夾")


def plot_confusion_matrices(conf_dir="../results/confusion matrix"):
    import glob

    pred_files = glob.glob(os.path.join(conf_dir, "*_predictions.csv"))

    for file in pred_files:
        if os.stat(file).st_size == 0:
            print(f"⚠️ 檢測到空檔案：{file}，已跳過")
            continue
        
        try:
            df = pd.read_csv(file)
        except pd.errors.EmptyDataError:
            print(f"⚠️ pandas 無法解析檔案（可能為空）：{file}")
            continue
        
        if not all(col in df.columns for col in ["Model", "y_test", "y_pred"]):
            print(f"⚠️ 欄位缺失，跳過：{file}")
            continue

        dataset_name = os.path.basename(file).replace("_predictions.csv", "").replace("_", " ").title()

        for model in df["Model"].unique():
            sub = df[df["Model"] == model]
            cm = confusion_matrix(sub["y_test"], sub["y_pred"])

            plt.figure(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', cbar=False)
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.title(f"{dataset_name} - {model}")
            plt.tight_layout()

            filename = f"{dataset_name}_{model}".replace(" ", "_").lower() + ".png"
            plt.savefig(os.path.join(conf_dir, filename))
            plt.close()

    print("✅ 混淆矩陣圖已儲存至 'confusion matrix' 資料夾")