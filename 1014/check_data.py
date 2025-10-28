import os
import glob

def check_data_locations():
    """檢查各資料夾的內容"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    folders_to_check = ['raw', 'cleaned', '0810/raw', '0810/cleaned', '../cleaned']
    
    print("📂 檢查資料位置：")
    print("=" * 50)
    
    for folder in folders_to_check:
        folder_path = os.path.join(base_dir, folder)
        abs_path = os.path.abspath(folder_path)
        
        print(f"\n📁 {folder}:")
        print(f"   路徑: {abs_path}")
        
        if os.path.exists(folder_path):
            # 檢查 CSV 檔案
            csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
            # 檢查 parquet 檔案
            parquet_files = glob.glob(os.path.join(folder_path, "*.parquet"))
            
            print(f"   ✅ 存在")
            print(f"   📄 CSV 檔案: {len(csv_files)}")
            print(f"   📦 Parquet 檔案: {len(parquet_files)}")
            
            # 顯示檔案名稱範例
            all_files = csv_files + parquet_files
            if all_files:
                print(f"   📋 檔案範例:")
                for i, file_path in enumerate(all_files[:3]):
                    filename = os.path.basename(file_path)
                    file_size = os.path.getsize(file_path) / (1024*1024)  # MB
                    print(f"      {i+1}. {filename} ({file_size:.1f} MB)")
                if len(all_files) > 3:
                    print(f"      ... 還有 {len(all_files)-3} 個檔案")
            else:
                print(f"   ⚠️ 沒有找到 CSV 或 Parquet 檔案")
        else:
            print(f"   ❌ 不存在")
    
    # 檢查上層目錄
    parent_dir = os.path.dirname(base_dir)
    print(f"\n🔍 檢查上層目錄: {parent_dir}")
    if os.path.exists(parent_dir):
        subdirs = [d for d in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, d))]
        print(f"   子目錄: {subdirs}")

if __name__ == "__main__":
    check_data_locations()