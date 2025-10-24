# -*- coding: utf-8 -*-
"""
交通流量預測專案 - 主程式
"""

import pandas as pd
import numpy as np
import os
import json
from config import config
from data_loader import DataLoader
from preprocess import DataPreprocessor
from feature_engineering import FeatureEngineer
from train_models import ModelTrainer
from evaluate import ModelEvaluator

class TrafficFlowPredictor:
    """交通流量預測主控制器"""
    
    def __init__(self):
        self.data_loader = DataLoader()
        self.preprocessor = DataPreprocessor()
        self.feature_engineer = FeatureEngineer()
        self.trainer = ModelTrainer()
        self.evaluator = ModelEvaluator()
        
    def run_complete_pipeline(self):
        """執行完整的預測流程"""
        print("=" * 60)
        print("🚦 交通流量預測專案")
        print("=" * 60)
        
        # 1. 載入資料
        print("\n📊 載入資料...")
        datasets = self.data_loader.load_all_datasets()
        
        if not datasets:
            print("❌ 未找到資料集，程式結束")
            return
        
        # 顯示資料集資訊
        dataset_info = self.data_loader.get_dataset_info()
        print(dataset_info)
        
        # 2. 對每個資料集進行處理和建模
        for dataset_name, raw_data in datasets.items():
            print(f"\n🔄 處理資料集: {dataset_name}")
            print("-" * 40)
            
            # 資料前處理
            print("🧹 資料清洗...")
            clean_data = self.preprocessor.clean_data(raw_data)
            
            # 特徵工程
            print("⚙️ 特徵工程...")
            engineered_data = self.feature_engineer.engineer_features(clean_data)
            
            # 準備特徵和目標變數
            feature_columns = [col for col in engineered_data.columns 
                             if col not in ['datetime', 'traffic_flow', 'dataset_source']]
            
            X = engineered_data[feature_columns].values
            y = engineered_data['traffic_flow'].values
            
            # 分割資料
            train_data, val_data, test_data = self.preprocessor.split_data(engineered_data)
            
            X_train = train_data[feature_columns].values
            y_train = train_data['traffic_flow'].values
            X_test = test_data[feature_columns].values
            y_test = test_data['traffic_flow'].values
            
            # 特徵標準化
            X_train_scaled, X_test_scaled = self.preprocessor.scale_features(
                X_train, X_test=X_test
            )
            
            # 模型訓練
            print("🤖 訓練模型...")
            self.trainer.train_all_models(X_train_scaled, y_train)
            
            # 模型評估
            print("📈 評估模型...")
            results = self.evaluator.evaluate_models_on_dataset(
                self.trainer, X_test_scaled, y_test, dataset_name
            )
            
            # 儲存模型
            self.trainer.save_models(dataset_name)
            
            print(f"✅ {dataset_name} 處理完成")
        
        # 3. 生成比較分析
        print("\n📊 生成比較分析...")
        self._generate_analysis()
        
        print("\n🎉 所有處理完成！")
        print(f"📁 結果儲存在: {config.RESULTS_DIR}")
    
    def _generate_analysis(self):
        """生成分析報告"""
        # 模型比較
        comparison_df = self.evaluator.compare_models()
        print("\n📋 模型比較結果:")
        print(comparison_df)
        
        # 生成比較圖表
        self.evaluator.generate_comparison_plots()
        
        # 生成性能矩陣
        matrix_df = self.evaluator.generate_performance_matrix()
        
        # 生成總結報告
        summary = self.evaluator.generate_summary_report()
        
        # 儲存總結報告
        summary_path = os.path.join(
            config.RESULTS_DIR,
            f"summary_report_{config.TIMESTAMP}.json"
        )
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        print(f"\n📄 總結報告已儲存: {summary_path}")
        
        # 顯示最佳結果
        if summary['overall_best']:
            best = summary['overall_best']
            print(f"\n🏆 最佳組合: {best['model']} + {best['dataset']}")
            print(f"   R² = {best['R2']:.4f}")

def main():
    """主函數"""
    try:
        predictor = TrafficFlowPredictor()
        predictor.run_complete_pipeline()
    except Exception as e:
        print(f"❌ 執行錯誤: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()