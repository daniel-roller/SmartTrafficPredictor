import os
import sys
import time
import subprocess

# 確保當前目錄在Python路徑中
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

def run_python_script(script_name, description):
    """使用subprocess執行Python腳本"""
    print(f"\n🚀 {description}...")
    start_time = time.time()
    
    try:
        script_path = os.path.join(current_dir, script_name)
        
        if not os.path.exists(script_path):
            print(f"❌ 找不到檔案: {script_path}")
            return 0
        
        # 使用subprocess執行腳本
        result = subprocess.run(
            [sys.executable, script_path], 
            cwd=current_dir,
            capture_output=True, 
            text=True, 
            encoding='utf-8'
        )
        
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ {description} 完成 ({elapsed:.1f}秒)")
            # 輸出腳本的標準輸出
            if result.stdout.strip():
                print("📋 執行結果:")
                print(result.stdout)
        else:
            print(f"❌ {description} 失敗 ({elapsed:.1f}秒)")
            print(f"錯誤輸出: {result.stderr}")
            if result.stdout.strip():
                print(f"標準輸出: {result.stdout}")
        
        return elapsed
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"❌ {description} 異常 ({elapsed:.1f}秒)")
        print(f"錯誤: {e}")
        return elapsed

def run_direct_import(script_name, description):
    """直接import並執行腳本"""
    print(f"\n🚀 {description}...")
    start_time = time.time()
    
    try:
        # 移除.py副檔名
        module_name = script_name.replace('.py', '')
        
        # 直接import模組
        module = __import__(module_name)
        
        # 執行main函數
        if hasattr(module, 'main'):
            module.main()
        else:
            print(f"⚠️ {module_name} 沒有main函數")
        
        elapsed = time.time() - start_time
        print(f"✅ {description} 完成 ({elapsed:.1f}秒)")
        return elapsed
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"❌ {description} 失敗 ({elapsed:.1f}秒)")
        print(f"錯誤: {e}")
        import traceback
        traceback.print_exc()
        return elapsed

def main():
    """主執行流程"""
    print("🚦 SmartTrafficPredictor 多執行緒完整流程")
    print("=" * 60)
    
    # 載入config以檢查路徑
    try:
        from config import Config
        config = Config()
    except Exception as e:
        print(f"❌ 無法載入配置: {e}")
        return
    
    total_start = time.time()
    
    # 檢查必要目錄和檔案
    raw_files = [f for f in os.listdir(config.RAW_DIR) if f.endswith('.csv')] if os.path.exists(config.RAW_DIR) else []
    cleaned_files = [f for f in os.listdir(config.CLEANED_DIR) if f.endswith('.parquet')] if os.path.exists(config.CLEANED_DIR) else []
    
    print(f"📂 原始資料: {len(raw_files)} 個CSV檔案")
    print(f"📂 清理資料: {len(cleaned_files)} 個Parquet檔案")
    
    if raw_files and not cleaned_files:
        print("💡 需要先清理原始資料")
        clean_time = run_direct_import("make_cleaned.py", "資料清理")
        # 重新檢查清理後的檔案
        cleaned_files = [f for f in os.listdir(config.CLEANED_DIR) if f.endswith('.parquet')] if os.path.exists(config.CLEANED_DIR) else []
        print(f"📦 清理完成，產生 {len(cleaned_files)} 個檔案")
    
    if not cleaned_files:
        print("❌ 沒有可用的清理資料，請檢查 raw/ 目錄或執行 make_cleaned.py")
        return
    
    # 執行流程
    steps = [
        ("segment_selector.py", "路段選擇與分析"),
        ("make_dataset_multithread.py", "多執行緒資料集建立"),
        ("train_models_multithread.py", "多執行緒模型訓練"),
        ("analyze_results.py", "結果分析與視覺化")
    ]
    
    step_times = []
    
    for script_file, description in steps:
        script_path = os.path.join(current_dir, script_file)
        if os.path.exists(script_path):
            step_time = run_direct_import(script_file, description)
            step_times.append((description, step_time))
        else:
            print(f"⚠️ 跳過 {script_file} (檔案不存在)")
            step_times.append((description, 0))
    
    # 總結
    total_time = time.time() - total_start
    
    print("\n" + "=" * 60)
    print("🎉 SmartTrafficPredictor 多執行緒流程完成！")
    print("=" * 60)
    print(f"⏱️ 時間統計:")
    for desc, step_time in step_times:
        print(f"  - {desc}: {step_time:.1f}秒")
    print(f"  - 總時間: {total_time:.1f}秒")
    
    # 檢查產生的檔案
    datasets = [f for f in os.listdir(config.DATASETS_DIR) if f.endswith('.pkl')] if os.path.exists(config.DATASETS_DIR) else []
    models = [f for f in os.listdir(config.MODELS_DIR) if f.endswith('.h5')] if os.path.exists(config.MODELS_DIR) else []
    results = [f for f in os.listdir(config.RESULTS_DIR)] if os.path.exists(config.RESULTS_DIR) else []
    
    print(f"\n📊 產生的檔案:")
    print(f"  - 資料集: {len(datasets)} 個 .pkl 檔案")
    print(f"  - 模型: {len(models)} 個 .h5 檔案")  
    print(f"  - 結果: {len(results)} 個檔案")
    
    print(f"\n📁 結果檔案位置:")
    print(f"  - 資料集: {config.DATASETS_DIR}")
    print(f"  - 模型: {config.MODELS_DIR}")
    print(f"  - 結果: {config.RESULTS_DIR}")

if __name__ == "__main__":
    main()