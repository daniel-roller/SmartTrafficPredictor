import os
import pandas as pd
import glob
from config import Config

def analyze_cleaned_data():
    """分析清理後的資料，挑選適合的路段"""
    config = Config()
    
    print(f"🔍 搜尋目錄: {config.CLEANED_DIR}")
    
    # 先嘗試parquet檔案
    parquet_files = glob.glob(os.path.join(config.CLEANED_DIR, "*.parquet"))
    csv_files = glob.glob(os.path.join(config.CLEANED_DIR, "*.csv"))
    
    all_files = parquet_files + csv_files
    print(f"� 找到 {len(all_files)} 個清理後的資料檔案 (parquet: {len(parquet_files)}, csv: {len(csv_files)})")
    
    if not all_files:
        raise FileNotFoundError(f"在 {config.CLEANED_DIR} 中找不到任何資料檔案")
    
    segments_info = []
    
    for file_path in all_files:
        try:
            # 根據副檔名選擇讀取方式
            if file_path.endswith('.parquet'):
                df = pd.read_parquet(file_path)
            else:
                df = pd.read_csv(file_path)
            
            segment_name = os.path.basename(file_path).replace('.parquet', '').replace('.csv', '')
            
            # 檢查並標準化欄位名稱
            df.columns = [col.strip().lower() for col in df.columns]
            
            # 尋找時間欄位
            time_col = None
            for col in df.columns:
                if 'time' in col or 'date' in col:
                    time_col = col
                    break
            
            # 尋找速度欄位
            speed_col = None
            for col in df.columns:
                if 'speed' in col:
                    speed_col = col
                    break
            
            if not time_col or not speed_col:
                print(f"⚠️ {segment_name} 缺少必要欄位 (time或speed)")
                continue
            
            # 確保時間欄位是datetime類型
            if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
                df[time_col] = pd.to_datetime(df[time_col])
            
            info = {
                'segment': segment_name,
                'total_records': len(df),
                'date_range_days': (df[time_col].max() - df[time_col].min()).days,
                'speed_range': f"{df[speed_col].min():.1f}~{df[speed_col].max():.1f}",
                'data_completeness': (df[speed_col].notna().sum() / len(df)) * 100,
                'file_path': file_path,
                'columns': list(df.columns),
                'time_col': time_col,
                'speed_col': speed_col
            }
            segments_info.append(info)
            
        except Exception as e:
            print(f"⚠️ 無法分析 {file_path}: {e}")
    
    if not segments_info:
        raise ValueError("沒有成功分析任何資料檔案")
    
    # 排序並挑選最適合的路段
    segments_df = pd.DataFrame(segments_info)
    segments_df = segments_df.sort_values(
        ['data_completeness', 'total_records'], 
        ascending=[False, False]
    )
    
    print("\n📊 路段資料分析結果：")
    display_cols = ['segment', 'total_records', 'date_range_days', 'data_completeness']
    print(segments_df[display_cols].to_string(index=False))
    
    # 挑選前3個最適合的路段（或全部如果少於3個）
    num_to_select = min(3, len(segments_df))
    selected_segments = segments_df.head(num_to_select)
    
    print(f"\n🎯 挑選的路段：")
    for idx, row in selected_segments.iterrows():
        print(f"  - {row['segment']}: {row['total_records']} 筆資料, 完整度 {row['data_completeness']:.1f}%")
        print(f"    時間欄位: {row['time_col']}, 速度欄位: {row['speed_col']}")
    
    return selected_segments['segment'].tolist(), selected_segments

def main():
    """主函數"""
    try:
        selected_names, selected_info = analyze_cleaned_data()
        print(f"\n✅ 成功選擇 {len(selected_names)} 個路段進行後續處理")
        return selected_names, selected_info
    except Exception as e:
        print(f"❌ 路段選擇失敗: {e}")
        import traceback
        traceback.print_exc()
        return [], None

if __name__ == "__main__":
    main()