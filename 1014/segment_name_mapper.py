import os
import pandas as pd
import glob
from config import Config

class SegmentNameMapper:
    """路段名稱映射器"""
    
    def __init__(self):
        self.mapping = {
            # 原始中文名稱 -> 英文名稱
            '國道1號_北向_五甲系統交流道_瑞隆路交流道': 'Highway1_North_WuJia_RuiLong',
            '國道3號_北向_中投交流道_烏日交流道': 'Highway3_North_ZhongTou_WuRi', 
            '國道5號_北向_彭山隧道前_彭山隧道後': 'Highway5_North_PengShan_Tunnel',
            '沙鹿架構_北向_中港路接一交流道': 'ShaLu_North_ZhongGang_Interchange'
        }
        
        # 反向映射
        self.reverse_mapping = {v: k for k, v in self.mapping.items()}
    
    def chinese_to_english(self, chinese_name):
        """中文轉英文"""
        return self.mapping.get(chinese_name, chinese_name.replace('_', '_'))
    
    def english_to_chinese(self, english_name):
        """英文轉中文"""
        return self.reverse_mapping.get(english_name, english_name)
    
    def rename_parquet_files(self):
        """重新命名parquet檔案為英文"""
        config = Config()
        parquet_files = glob.glob(os.path.join(config.CLEANED_DIR, "*.parquet"))
        
        renamed_count = 0
        for file_path in parquet_files:
            filename = os.path.basename(file_path)
            name_without_ext = filename.replace('.parquet', '')
            
            if name_without_ext in self.mapping:
                new_name = self.mapping[name_without_ext] + '.parquet'
                new_path = os.path.join(config.CLEANED_DIR, new_name)
                
                os.rename(file_path, new_path)
                print(f"✅ 重新命名: {filename} -> {new_name}")
                renamed_count += 1
        
        print(f"🔄 總共重新命名 {renamed_count} 個檔案")
        return renamed_count

def main():
    mapper = SegmentNameMapper()
    mapper.rename_parquet_files()

if __name__ == "__main__":
    main()