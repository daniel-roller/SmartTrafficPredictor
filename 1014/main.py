# -*- coding: utf-8 -*-
"""
交通流量預測系統 - 主程式
一鍵執行完整的傳統機器學習實驗流程
"""

import os
import sys
import warnings
import traceback
from datetime import datetime

# 添加當前目錄到路徑
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 導入自定義模組
from config import config
from data_processor import DataProcessor
from traditional_models import TraditionalMLModels
from evaluator import ModelEvaluator
from utils import (
    print_section_header, create_experiment_directory, 
    log_experiment_info, Timer, memory_usage
)

# 忽略警告
warnings.filterwarnings('ignore')

class ExperimentRunner:
    """實驗執行器"""
    
    def __init__(self):
        self.timer = Timer()
        self.experiment_dir = None
        self.data_processor = DataProcessor()
        self.ml_trainer = TraditionalMLModels()
        self.evaluator = ModelEvaluator()
        
    def setup_experiment(self) -> str:
        """設定實驗環境"""
        print_section_header("🚀 交通流量預測 - 傳統機器學習實驗", width=80)
        
        # 建立實驗資料夾
        self.experiment_dir = create_experiment_directory()
        print(f"📁 實驗資料夾: {self.experiment_dir}")
        
        # 記錄實驗資訊
        experiment_info = {
            'experiment_type': '傳統機器學習',
            'datasets': list(config.DATASETS_INFO.keys()),
            'models': list(config.TRADITIONAL_ML_PARAMS.keys()),
            'feature_engineering': True,
            'cross_validation': True,
            'memory_usage_start': memory_usage()
        }
        
        log_experiment_info(self.experiment_dir, experiment_info)
        
        print(f"🔧 設定概況:")
        print(f"  - 資料集: {len(config.DATASETS_INFO)} 個")
        print(f"  - 模型: {len(config.TRADITIONAL_ML_PARAMS)} 個")
        print(f"  - 訓練比例: {config.TRAIN_RATIO*100:.0f}%")
        print(f"  - 驗證比例: {config.VAL_RATIO*100:.0f}%")
        print(f"  - 測試比例: {config.TEST_RATIO*100:.0f}%")
        print(f"  - 交叉驗證: {config.CV_FOLDS} 折")
        print(f"  - 最大樣本數: {config.MAX_SAMPLE_SIZE}")
        print(f"  - 記憶體使用: {memory_usage()}")
        
        return self.experiment_dir
    
    def run_data_processing(self) -> dict:
        """執行資料處理"""
        print_section_header("📊 資料載入與預處理階段", width=80)
        
        try:
            # 載入資料集
            self.timer.start()
            raw_datasets = self.data_processor.load_all_datasets()
            
            if not raw_datasets:
                raise Exception("沒有成功載入任何資料集")
            
            # 資料預處理
            processed_datasets = self.data_processor.process_all_datasets()
            
            if not processed_datasets:
                raise Exception("資料預處理失敗")
            
            self.timer.stop()
            
            print(f"\n✅ 資料處理完成 ({self.timer.elapsed_str()})")
            print(f"📈 處理後資料集統計:")
            
            for name, data in processed_datasets.items():
                print(f"  - {name}: {data['feature_count']} 特徵, "
                      f"{len(data['X_train'])+len(data['X_val'])+len(data['X_test'])} 樣本")
            
            return processed_datasets
            
        except Exception as e:
            print(f"❌ 資料處理失敗: {e}")
            traceback.print_exc()
            return {}
    
    def run_model_training(self, processed_datasets: dict) -> dict:
        """執行模型訓練"""
        print_section_header("🤖 機器學習模型訓練階段", width=80)
        
        all_results = {}
        
        for dataset_name, processed_data in processed_datasets.items():
            try:
                print(f"\n{'='*60}")
                print(f"🔄 開始處理資料集: {dataset_name}")
                print(f"{'='*60}")
                
                # 訓練模型
                self.timer.start()
                dataset_results = self.ml_trainer.train_all_models(processed_data)
                self.timer.stop()
                
                all_results[dataset_name] = dataset_results
                
                # 顯示該資料集的簡要結果
                successful_models = sum(1 for r in dataset_results.values() 
                                      if r.get('test_metrics'))
                
                print(f"\n📊 {dataset_name} 結果:")
                print(f"  ✅ 成功訓練: {successful_models}/{len(dataset_results)} 個模型")
                print(f"  ⏱️ 訓練耗時: {self.timer.elapsed_str()}")
                
                if successful_models > 0:
                    best_model = max(
                        [name for name in dataset_results.keys() 
                         if dataset_results[name].get('test_metrics')],
                        key=lambda x: dataset_results[x]['test_metrics']['R²']
                    )
                    best_r2 = dataset_results[best_model]['test_metrics']['R²']
                    print(f"  🏆 最佳模型: {best_model} (R² = {best_r2:.4f})")
                
            except Exception as e:
                print(f"❌ 處理 {dataset_name} 失敗: {e}")
                traceback.print_exc()
                all_results[dataset_name] = {}
        
        return all_results
    
    def run_evaluation(self, all_results: dict) -> None:
        """執行模型評估"""
        print_section_header("📈 模型評估與結果分析階段", width=80)
        
        try:
            self.timer.start()
            
            # 設定結果儲存路徑
            results_dir = os.path.join(self.experiment_dir, 'results') if self.experiment_dir else config.RESULTS_DIR
            plots_dir = os.path.join(self.experiment_dir, 'plots') if self.experiment_dir else config.PLOTS_DIR
            
            os.makedirs(results_dir, exist_ok=True)
            os.makedirs(plots_dir, exist_ok=True)
            
            # 執行評估
            self.evaluator.evaluate_all_results(all_results)
            
            self.timer.stop()
            print(f"\n✅ 評估完成 ({self.timer.elapsed_str()})")
            
        except Exception as e:
            print(f"❌ 評估失敗: {e}")
            traceback.print_exc()
    
    def run_full_experiment(self) -> bool:
        """執行完整實驗流程"""
        total_timer = Timer()
        total_timer.start()
        
        try:
            # 1. 設定實驗
            self.setup_experiment()
            
            # 2. 資料處理
            processed_datasets = self.run_data_processing()
            if not processed_datasets:
                print("❌ 實驗中止：資料處理失敗")
                return False
            
            # 3. 模型訓練
            all_results = self.run_model_training(processed_datasets)
            if not any(results for results in all_results.values()):
                print("❌ 實驗中止：模型訓練失敗")
                return False
            
            # 4. 評估結果
            self.run_evaluation(all_results)
            
            # 5. 實驗總結
            total_timer.stop()
            
            print_section_header("🎉 實驗完成總結", width=80)
            print(f"⏰ 總執行時間: {total_timer.elapsed_str()}")
            print(f"💾 記憶體使用: {memory_usage()}")
            
            if self.experiment_dir:
                print(f"📁 結果儲存於: {self.experiment_dir}")
            
            # 統計成功率
            total_experiments = sum(len(results) for results in all_results.values())
            successful_experiments = sum(
                sum(1 for r in results.values() if r.get('test_metrics'))
                for results in all_results.values()
            )
            
            success_rate = successful_experiments / total_experiments * 100 if total_experiments > 0 else 0
            
            print(f"\n📊 實驗統計:")
            print(f"  - 資料集數量: {len(processed_datasets)}")
            print(f"  - 模型實驗數: {total_experiments}")
            print(f"  - 成功實驗數: {successful_experiments}")
            print(f"  - 成功率: {success_rate:.1f}%")
            
            # 最佳模型摘要
            print(f"\n🏆 各資料集最佳模型:")
            for dataset_name, results in all_results.items():
                valid_results = {name: result for name, result in results.items() 
                               if result.get('test_metrics')}
                if valid_results:
                    best_model = max(valid_results.keys(), 
                                   key=lambda x: valid_results[x]['test_metrics']['R²'])
                    best_r2 = valid_results[best_model]['test_metrics']['R²']
                    best_rmse = valid_results[best_model]['test_metrics']['RMSE']
                    print(f"  - {dataset_name}: {best_model} "
                          f"(R² = {best_r2:.4f}, RMSE = {best_rmse:.4f})")
            
            print(f"\n🎯 實驗成功完成！")
            print(f"📈 詳細結果請查看生成的圖表和報告檔案")
            
            return True
            
        except Exception as e:
            total_timer.stop()
            print(f"\n❌ 實驗執行失敗: {e}")
            print(f"⏰ 執行時間: {total_timer.elapsed_str()}")
            traceback.print_exc()
            return False

def main():
    """主函數"""
    print(f"""
    🚗 交通流量預測系統
    ==========================================
    版本: 1.0
    功能: 傳統機器學習模型比較實驗
    時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    ==========================================
    """)
    
    # 檢查必要檔案
    if not os.path.exists(config.DATASETS_DIR):
        print(f"❌ 找不到資料集資料夾: {config.DATASETS_DIR}")
        print("請確認資料集檔案已放置在正確位置")
        return
    
    # 建立必要資料夾
    config.create_directories()
    
    # 執行實驗
    runner = ExperimentRunner()
    success = runner.run_full_experiment()
    
    if success:
        print("\n🎊 實驗圓滿完成！")
        if runner.experiment_dir:
            print(f"📂 所有結果已儲存至: {runner.experiment_dir}")
        print("\n感謝使用交通流量預測系統！")
    else:
        print("\n💥 實驗未能完成，請檢查錯誤訊息並重試")
        sys.exit(1)

if __name__ == "__main__":
    main()