# main.py
# === 全流程自動化版 ===
# 功能：依序執行 make_dataset、train_lstm、predict、baseline、train_prophet_light
# 可選只跑 Prophet 或全部流程

import subprocess
import os
import time
import re

RESULTS_DIR = "../results"

def run_step(name, command):
    """執行每個步驟"""
    print(f"\n===== 開始執行：{name} =====")
    start = time.time()
    result = subprocess.run(command, shell=True)
    end = time.time()
    elapsed = end - start
    if result.returncode == 0:
        print(f"✅ {name} 完成，用時 {elapsed:.2f} 秒")
    else:
        print(f"❌ {name} 執行失敗，請檢查錯誤訊息")
        exit(1)
    return elapsed

def parse_lstm_metrics():
    """讀取 LSTM 的測試誤差"""
    metrics_path = os.path.join(RESULTS_DIR, "metrics_test.txt")
    mae, mape = None, None
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            text = f.read()
        mae_match = re.search(r"MAE:\s*([\d.]+)", text)
        mape_match = re.search(r"MAPE:\s*([\d.]+)%", text)
        if mae_match: mae = float(mae_match.group(1))
        if mape_match: mape = float(mape_match.group(1))
    return mae, mape


if __name__ == "__main__":
    print("🚦 智慧交通預測系統 - 自動流程啟動")

    # === 讓使用者決定要跑哪一種 ===
    mode = input("\n請選擇模式：\n"
                 "1️⃣ 短期 + 長期 (完整流程)\n"
                 "2️⃣ 只跑 Prophet 長期預測 (快速模式)\n"
                 "輸入 1 或 2: ").strip()

    total_start = time.time()

    # === 1️⃣ 全部流程 ===
    if mode == "1":
        print("\n🚀 執行完整流程：LSTM（短期）+ Prophet（長期）")

        total_elapsed = 0
        total_elapsed += run_step("資料集製作 (make_dataset)", "python make_dataset.py")
        total_elapsed += run_step("LSTM 模型訓練", "python train_lstm.py")
        total_elapsed += run_step("LSTM 模型預測", "python predict.py")
        total_elapsed += run_step("Baseline 比較", "python baseline.py")
        total_elapsed += run_step("Prophet 長期趨勢預測", "python train_prophet_light.py")

        # 讀 LSTM 評估
        mae, mape = parse_lstm_metrics()
        summary_path = os.path.join(RESULTS_DIR, "run_summary.txt")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(f"LSTM Test MAE={mae}, MAPE={mape}%\n")
            f.write(f"Total time: {total_elapsed:.2f} sec\n")

        print(f"\n✅ 全部流程完成，用時 {total_elapsed:.2f} 秒")
        print(f"📄 結果已存到: {summary_path}")

    # === 2️⃣ 只跑 Prophet ===
    elif mode == "2":
        print("\n📆 只執行 Prophet 長期預測（30 天）")
        total_elapsed = 0
        total_elapsed += run_step("資料集製作 (make_dataset)", "python make_dataset.py")
        total_elapsed += run_step("Prophet 長期趨勢預測", "python train_prophet_light.py")

        print(f"\n✅ Prophet 預測完成，用時 {total_elapsed:.2f} 秒")
    else:
        print("⚠️ 無效輸入，請重新執行並輸入 1 或 2。")
        exit(0)

    total_end = time.time()
    print(f"\n🎉 全部任務完成，總耗時 {total_end - total_start:.2f} 秒")

    print(f"\n📊 結果檔案可在 results/ 目錄下查看：")
    print(" - pred_vs_true_LSTM.png：短期速度預測圖")
    print(" - prophet_30d_fixed/prophet_forecast_30d_fixed.png：長期趨勢圖")
    print(" - metrics_prophet_30d_fixed.txt：Prophet 回測指標")
    print(" - metrics_test.txt：LSTM 預測誤差")
