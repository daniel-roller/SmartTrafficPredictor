# main.py
import subprocess
import os
import re

RESULTS_DIR = "../results"  # results 放在 scripts 的上層

def run_step(name, command):
    print(f"\n===== 開始執行：{name} =====")
    result = subprocess.run(command, shell=True)
    if result.returncode == 0:
        print(f"✅ {name} 完成")
    else:
        print(f"❌ {name} 執行失敗，請檢查錯誤訊息")
        exit(1)
    
def update_config(horizon):
    """修改同資料夾下 config.py 的 HORIZON 值"""
    config_path = "config.py"
    with open(config_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    with open(config_path, "w", encoding="utf-8") as f:
        for line in lines:
            if line.strip().startswith("HORIZON"):
                f.write(f"HORIZON = {horizon}\n")
            else:
                f.write(line)

def parse_metrics():
    """讀取 metrics_test.txt 的數值"""
    metrics_file = os.path.join(RESULTS_DIR, "metrics_test.txt")
    mae, mape = None, None
    if os.path.exists(metrics_file):
        with open(metrics_file, "r", encoding="utf-8") as f:
            text = f.read()
        mae_match = re.search(r"MAE:\s*([\d.]+)", text)
        mape_match = re.search(r"MAPE:\s*([\d.]+)%", text)
        if mae_match: mae = float(mae_match.group(1))
        if mape_match: mape = float(mape_match.group(1))
    return mae, mape

if __name__ == "__main__":
    # 你要測試的 HORIZON 值（單位=15分鐘）
    horizons = [1, 2, 4, 6, 12, 24, 48, 96]

    results = []

    for h in horizons:
        print(f"\n🚀 開始實驗：HORIZON={h}")
        update_config(h)

        # 跑 pipeline (都在同資料夾下)
        run_step("資料集製作 (make_dataset)", "python make_dataset.py")
        run_step("模型訓練 (train_lstm)", "python train_lstm.py")
        run_step("模型預測 (predict)", "python predict.py")
        run_step("Baseline 比較 (baseline)", "python baseline.py")
        run_step("Prophet 長期預測", "python train_prophet_light.py")


        # 收集指標
        mae, mape = parse_metrics()
        results.append({"HORIZON": h, "Test MAE": mae, "Test MAPE": mape})

    # 印出比較結果
    print("\n📊 各 HORIZON 測試結果比較：")
    summary_lines = []
    for r in results:
        line = f"HORIZON={r['HORIZON']}: MAE={r['Test MAE']}, MAPE={r['Test MAPE']}%"
        print(line)
        summary_lines.append(line)

    # 存成檔案
    os.makedirs(RESULTS_DIR, exist_ok=True)
    summary_path = os.path.join(RESULTS_DIR, "horizon_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

    print("\n🎉 所有實驗完成！結果已存到:", summary_path)
