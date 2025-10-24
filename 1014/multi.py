# -*- coding: utf-8 -*-
"""
多執行緒交通流量預測專案 - 主程式 (優化版)
"""

import pandas as pd
import numpy as np
import os
import json
import concurrent.futures
import threading
import time
from datetime import datetime

# 確保所有匯入都正確
try:
    from config import config
    print("✅ config 匯入成功")
except ImportError as e:
    print(f"❌ config 匯入失敗: {e}")

try:
    from data_loader import DataLoader
    print("✅ DataLoader 匯入成功")
except ImportError as e:
    print(f"❌ DataLoader 匯入失敗: {e}")

try:
    from preprocess import DataPreprocessor
    print("✅ DataPreprocessor 匯入成功")
except ImportError as e:
    print(f"❌ DataPreprocessor 匯入失敗: {e}")

try:
    from feature_engineering import FeatureEngineer
    print("✅ FeatureEngineer 匯入成功")
except ImportError as e:
    print(f"❌ FeatureEngineer 匯入失敗: {e}")

try:
    from train_models import MultiThreadModelTrainer
    print("✅ MultiThreadModelTrainer 匯入成功")
except ImportError as e:
    print(f"❌ MultiThreadModelTrainer 匯入失敗: {e}")

try:
    from evaluate import ModelEvaluator
    print("✅ ModelEvaluator 匯入成功")
except ImportError as e:
    print(f"❌ ModelEvaluator 匯入失敗: {e}")

try:
    from multithread_utils import ProgressTracker, ResourceMonitor, parallel_execute
    print("✅ multithread_utils 匯入成功")
except ImportError as e:
    print(f"❌ multithread_utils 匯入失敗: {e}")

class MultiThreadTrafficPredictor:
    """多執行緒交通流量預測器 (優化版)"""
    
    def __init__(self, fast_mode=True):
        self.lock = threading.Lock()
        self.results = {}
        self.resource_monitor = ResourceMonitor()
        self.fast_mode = fast_mode
        
    def process_single_dataset(self, dataset_item):
        """處理單一資料集（執行緒安全 + 優化版）"""
        dataset_name, raw_data = dataset_item
        thread_id = threading.current_thread().name
        
        with self.lock:
            mode_text = "快速" if self.fast_mode else "完整"
            print(f"🔄 [{thread_id}] 開始處理資料集: {dataset_name} ({mode_text}模式)")
        
        try:
            # 建立獨立的處理器實例
            preprocessor = DataPreprocessor()
            feature_engineer = FeatureEngineer()
            trainer = MultiThreadModelTrainer()
            
            # 資料前處理
            with self.lock:
                print(f"🧹 [{thread_id}] 資料清洗: {dataset_name}")
            clean_data = preprocessor.clean_data(raw_data)
            
            # 特徵工程
            with self.lock:
                print(f"⚙️ [{thread_id}] 特徵工程: {dataset_name}")
            engineered_data = feature_engineer.engineer_features(clean_data)
            
            # 準備訓練資料
            feature_columns = [col for col in engineered_data.columns 
                             if col not in ['datetime', 'traffic_flow', 'dataset_source']]
            
            with self.lock:
                print(f"📊 [{thread_id}] 特徵數量: {len(feature_columns)}")
            
            train_data, val_data, test_data = preprocessor.split_data(engineered_data)
            
            X_train = train_data[feature_columns].values
            y_train = train_data['traffic_flow'].values
            X_test = test_data[feature_columns].values
            y_test = test_data['traffic_flow'].values
            
            # 特徵標準化
            X_train_scaled, X_test_scaled = preprocessor.scale_features(
                X_train, X_test=X_test
            )
            
            # 多執行緒模型訓練
            with self.lock:
                print(f"🤖 [{thread_id}] 開始快速模型訓練: {dataset_name}")
                print(f"   📈 訓練資料: {len(X_train_scaled)} 筆")
                print(f"   📊 測試資料: {len(X_test_scaled)} 筆")
            
            model_results = trainer.train_models_parallel(
                X_train_scaled, y_train, X_test_scaled, y_test, dataset_name
            )
            
            # 儲存結果
            with self.lock:
                self.results[dataset_name] = model_results
                print(f"✅ [{thread_id}] 完成資料集處理: {dataset_name}")
                # 顯示結果摘要
                for model_name, metrics in model_results.items():
                    print(f"   📊 {model_name}: R² = {metrics['R2']:.4f}")
            
            return dataset_name, model_results
            
        except Exception as e:
            with self.lock:
                print(f"❌ [{thread_id}] 處理 {dataset_name} 時發生錯誤: {e}")
                import traceback
                traceback.print_exc()
            return dataset_name, None
    
    def run_parallel_pipeline(self):
        """執行完整的多執行緒流程 (優化版)"""
        start_time = time.time()
        
        print("=" * 60)
        print("🚦 多執行緒交通流量預測專案 (優化版)")
        print("=" * 60)
        print(f"⚡ 快速模式: {'開啟' if self.fast_mode else '關閉'}")
        print(f"🔧 資料集執行緒: {config.MAX_WORKERS_DATASETS} 個")
        print(f"🔧 模型訓練執行緒: {config.MAX_WORKERS_MODELS} 個")
        print(f"🤖 訓練模型類型: {list(config.MODELS.keys())}")
        
        # 啟動資源監控
        self.resource_monitor.start_monitoring()
        
        try:
            # 載入資料
            print("\n📊 載入資料...")
            data_loader = DataLoader()
            
            if self.fast_mode:
                datasets = data_loader.load_all_datasets(sample_size=config.SAMPLE_SIZE)
            else:
                datasets = data_loader.load_all_datasets()
            
            if not datasets:
                print("❌ 未找到資料集，程式結束")
                return
            
            # 顯示資料集資訊
            dataset_info = data_loader.get_dataset_info()
            print(dataset_info)

            print(f"\n🚀 開始多執行緒處理 {len(datasets)} 個資料集...")

            # 建立進度追蹤器
            progress = ProgressTracker(len(datasets))
            
            # 使用執行緒池處理資料集
            with concurrent.futures.ThreadPoolExecutor(max_workers=config.MAX_WORKERS_DATASETS) as executor:
                # 提交所有任務
                future_to_dataset = {
                    executor.submit(self.process_single_dataset, item): item[0]
                    for item in datasets.items()
                }
                
                # 收集結果
                completed_count = 0
                for future in concurrent.futures.as_completed(future_to_dataset, timeout=600):  # 10分鐘超時
                    dataset_name = future_to_dataset[future]
                    try:
                        result = future.result()
                        completed_count += 1
                        progress.update_progress(f"完成 {dataset_name} ({completed_count}/{len(datasets)})")
                    except Exception as e:
                        print(f"❌ 處理 {dataset_name} 時發生例外: {e}")
            
            # 生成比較分析
            if self.results:
                print("\n📊 生成比較分析...")
                self.generate_analysis()
            else:
                print("\n❌ 沒有成功的訓練結果")
            
        finally:
            # 停止資源監控
            self.resource_monitor.stop_monitoring()
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        print(f"\n🎉 多執行緒處理完成！")
        print(f"⏱️  總執行時間: {elapsed_time:.2f} 秒")
        print(f"📁 結果儲存在: {config.RESULTS_DIR}")
        
        # 效能報告
        self.print_performance_report(elapsed_time, len(datasets))
    
    def generate_analysis(self):
        """生成分析報告"""
        try:
            evaluator = ModelEvaluator()
            evaluator.results = self.results
            
            # 模型比較
            comparison_df = evaluator.compare_models()
            print("\n📋 模型比較結果:")
            print(comparison_df)
            
            # 生成圖表
            try:
                evaluator.generate_comparison_plots()
                print("📊 比較圖表已生成")
            except Exception as e:
                print(f"⚠️  圖表生成失敗: {e}")
            
            # 儲存總結報告
            summary = evaluator.generate_summary_report()
            summary_path = os.path.join(
                config.RESULTS_DIR,
                f"multithread_summary_{config.TIMESTAMP}.json"
            )
            
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            
            print(f"\n📄 總結報告: {summary_path}")
            
            # 顯示最佳結果
            if summary.get('overall_best'):
                best = summary['overall_best']
                print(f"\n🏆 最佳組合: {best['model']} + {best['dataset']}")
                print(f"   R² = {best['R2']:.4f}")
            
            # 顯示所有結果摘要
            print(f"\n📈 完整結果摘要:")
            for dataset_name, dataset_results in self.results.items():
                print(f"📊 {dataset_name}:")
                for model_name, metrics in dataset_results.items():
                    print(f"   {model_name}: R²={metrics['R2']:.4f}, RMSE={metrics['RMSE']:.2f}")
                
        except Exception as e:
            print(f"❌ 生成分析報告時發生錯誤: {e}")
            import traceback
            traceback.print_exc()
    
    def print_performance_report(self, elapsed_time, num_datasets):
        """印出效能報告"""
        print("\n" + "=" * 50)
        print("📊 效能報告")
        print("=" * 50)
        print(f"📈 處理資料集數量: {num_datasets}")
        print(f"🤖 訓練模型數量: {num_datasets * len(config.MODELS)}")  # 修正: 使用實際模型數量
        print(f"⏱️  總執行時間: {elapsed_time:.2f} 秒")
        print(f"📊 平均每個資料集: {elapsed_time/num_datasets:.2f} 秒")
        print(f"🚀 多執行緒加速比: ~{config.MAX_WORKERS_DATASETS:.1f}x")
        
        # 額外資訊
        if self.fast_mode:
            print(f"⚡ 快速模式效能:")
            print(f"   📊 每資料集樣本數: {config.SAMPLE_SIZE}")
            print(f"   🌳 每模型樹數量: {config.MODELS['XGBoost']['n_estimators']}")
        
        print("=" * 50)

def main():
    """主函數"""
    try:
        # 可以選擇快速模式或完整模式
        fast_mode = True  # 設為 False 可使用完整模式
        
        predictor = MultiThreadTrafficPredictor(fast_mode=fast_mode)
        predictor.run_parallel_pipeline()
        
    except KeyboardInterrupt:
        print("\n⏹️  使用者中斷執行")
    except Exception as e:
        print(f"❌ 執行錯誤: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()