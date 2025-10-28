import os
import pandas as pd
import glob
from config import Config

def comprehensive_data_check():
    """全面檢查資料完整性"""
    config = Config()
    
    print("🔍 SmartTrafficPredictor 資料完整性檢查")
    print("=" * 60)
    
    # 1. 檢查raw資料夾
    print("\n📁 檢查原始資料 (raw/):")
    raw_files = glob.glob(os.path.join(config.RAW_DIR, "*.csv"))
    print(f"   找到 CSV 檔案: {len(raw_files)}")
    
    if raw_files:
        print("   檔案清單:")
        for i, file_path in enumerate(raw_files[:5], 1):
            filename = os.path.basename(file_path)
            size_mb = os.path.getsize(file_path) / (1024*1024)
            print(f"     {i}. {filename} ({size_mb:.1f} MB)")
        
        if len(raw_files) > 5:
            print(f"     ... 還有 {len(raw_files)-5} 個檔案")
        
        # 檢查第一個檔案的結構
        sample_file = raw_files[0]
        print(f"\n   📋 樣本檔案結構 ({os.path.basename(sample_file)}):")
        try:
            df_sample = pd.read_csv(sample_file, nrows=5)
            print(f"     欄位: {list(df_sample.columns)}")
            print(f"     資料筆數: {len(pd.read_csv(sample_file))}")
            print(f"     時間範圍: {df_sample.iloc[0]['dataTime']} ~ {df_sample.iloc[-1]['dataTime']}")
        except Exception as e:
            print(f"     ⚠️ 無法讀取: {e}")
    
    # 2. 檢查cleaned資料夾
    print(f"\n📁 檢查清理資料 (cleaned/):")
    cleaned_files = glob.glob(os.path.join(config.CLEANED_DIR, "*.parquet"))
    print(f"   找到 Parquet 檔案: {len(cleaned_files)}")
    
    if cleaned_files:
        total_records = 0
        for file_path in cleaned_files:
            try:
                df = pd.read_parquet(file_path)
                total_records += len(df)
            except:
                pass
        print(f"   總資料筆數: {total_records:,}")
    
    # 3. 檢查datasets資料夾
    print(f"\n📁 檢查資料集 (datasets/):")
    dataset_files = glob.glob(os.path.join(config.DATASETS_DIR, "*.pkl"))
    print(f"   找到 PKL 檔案: {len(dataset_files)}")
    
    # 4. 檢查models資料夾
    print(f"\n📁 檢查模型 (models/):")
    model_files = glob.glob(os.path.join(config.MODELS_DIR, "*.h5"))
    print(f"   找到 H5 模型檔案: {len(model_files)}")
    
    # 5. 檢查results資料夾
    print(f"\n📁 檢查結果 (results/):")
    result_files = glob.glob(os.path.join(config.RESULTS_DIR, "*"))
    print(f"   找到結果檔案: {len(result_files)}")
    
    # 6. 建議下一步
    print(f"\n💡 建議執行順序:")
    if not raw_files:
        print("   ❌ 缺少原始資料 - 請將 CSV 檔案放入 raw/ 資料夾")
    elif not cleaned_files:
        print("   1️⃣ 執行 make_cleaned.py - 清理原始資料")
    elif not dataset_files:
        print("   2️⃣ 執行 make_dataset_multithread.py - 建立訓練資料集")
    elif not model_files:
        print("   3️⃣ 執行 train_models_multithread.py - 訓練模型")
    elif not result_files:
        print("   4️⃣ 執行 analyze_results.py - 分析結果")
    else:
        print("   ✅ 所有步驟完成 - 可以執行 main_multithread.py 查看整體流程")

if __name__ == "__main__":
    comprehensive_data_check()