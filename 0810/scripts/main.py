import subprocess

def run_step(name, command):
    print(f"\n===== 開始執行：{name} =====")
    result = subprocess.run(command, shell=True)
    if result.returncode == 0:
        print(f"✅ {name} 完成")
    else:
        print(f"❌ {name} 執行失敗，請檢查錯誤訊息")
        exit(1)

if __name__ == "__main__":
    # 1. 製作資料集
    run_step("資料集製作 (make_dataset)", "python make_dataset.py")

    # 2. 訓練 LSTM 模型
    run_step("模型訓練 (train_lstm)", "python train_lstm.py")

    # 3. 模型預測
    run_step("模型預測 (predict)", "python predict.py")

    # 4. Baseline 比較
    run_step("Baseline 比較 (baseline)", "python baseline.py")

    print("\n🎉 Pipeline 全部流程完成！")
