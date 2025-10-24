# -*- coding: utf-8 -*-
"""
模型評估與比較模組 (增強版)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any
import os
import json
from config import config
from utils import calculate_metrics, save_results

class ModelEvaluator:
    """增強版模型評估器"""
    
    def __init__(self):
        self.results = {}
        self.comparison_results = {}
        print("🔍 ModelEvaluator 初始化完成")
        
    def evaluate_models_on_dataset(self, trainer, X_test: np.ndarray, y_test: np.ndarray, 
                                 dataset_name: str) -> Dict[str, Dict[str, float]]:
        """在單一資料集上評估所有模型"""
        results = {}
        
        for model_name in trainer.trained_models.keys():
            y_pred = trainer.predict(model_name, X_test)
            metrics = calculate_metrics(y_test, y_pred)
            results[model_name] = metrics
            
            # 儲存詳細預測結果
            pred_df = pd.DataFrame({
                'actual': y_test,
                'predicted': y_pred,
                'residual': y_test - y_pred,
                'absolute_error': np.abs(y_test - y_pred),
                'percentage_error': np.abs((y_test - y_pred) / y_test) * 100,
                'dataset': dataset_name,
                'model': model_name
            })
            
            pred_path = os.path.join(
                config.RESULTS_DIR, 
                "predictions",
                f"{model_name}_{dataset_name}_detailed_predictions.csv"
            )
            save_results(pred_df, pred_path)
        
        self.results[dataset_name] = results
        return results
    
    def compare_models(self, dataset_name: str = None) -> pd.DataFrame:
        """比較不同模型的性能 (增強版)"""
        if dataset_name:
            results_to_compare = {dataset_name: self.results[dataset_name]}
        else:
            results_to_compare = self.results
        
        comparison_data = []
        
        for ds_name, ds_results in results_to_compare.items():
            for model_name, metrics in ds_results.items():
                row = {'Dataset': ds_name, 'Model': model_name}
                row.update(metrics)
                # 新增排名資訊
                row['R2_rank'] = 0
                row['RMSE_rank'] = 0
                comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # 計算排名
        if len(comparison_df) > 0:
            comparison_df['R2_rank'] = comparison_df.groupby('Dataset')['R2'].rank(ascending=False)
            comparison_df['RMSE_rank'] = comparison_df.groupby('Dataset')['RMSE'].rank(ascending=True)
            comparison_df['Average_rank'] = (comparison_df['R2_rank'] + comparison_df['RMSE_rank']) / 2
        
        # 儲存比較結果
        comparison_path = os.path.join(
            config.RESULTS_DIR,
            "metrics",
            f"enhanced_model_comparison_{config.TIMESTAMP}.csv"
        )
        save_results(comparison_df, comparison_path)
        
        return comparison_df
    
    def generate_detailed_analysis(self):
        """生成詳細分析報告"""
        analysis = {}
        
        # 模型整體表現統計
        all_results = []
        for dataset_name, results in self.results.items():
            for model_name, metrics in results.items():
                all_results.append({
                    'dataset': dataset_name,
                    'model': model_name,
                    **metrics
                })
        
        if all_results:
            results_df = pd.DataFrame(all_results)
            
            # 各模型平均表現
            model_avg = results_df.groupby('model')[config.COMPARISON_METRICS].mean()
            analysis['model_averages'] = model_avg.to_dict()
            
            # 各資料集平均表現
            dataset_avg = results_df.groupby('dataset')[config.COMPARISON_METRICS].mean()
            analysis['dataset_averages'] = dataset_avg.to_dict()
            
            # 模型穩定性（標準差）
            model_std = results_df.groupby('model')[config.COMPARISON_METRICS].std()
            analysis['model_stability'] = model_std.to_dict()
            
            # 最佳模型統計
            best_models = {}
            for metric in config.COMPARISON_METRICS:
                if metric in ['R2']:  # 越大越好
                    best_model = results_df.loc[results_df[metric].idxmax(), 'model']
                else:  # 越小越好
                    best_model = results_df.loc[results_df[metric].idxmin(), 'model']
                best_models[metric] = best_model
            analysis['best_models_by_metric'] = best_models
        
        # 儲存詳細分析
        analysis_path = os.path.join(
            config.RESULTS_DIR,
            "analysis",
            f"detailed_analysis_{config.TIMESTAMP}.json"
        )
        
        with open(analysis_path, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, ensure_ascii=False, indent=2)
        
        print(f"📊 詳細分析報告: {analysis_path}")
        return analysis
    
    def generate_comparison_plots(self):
        """生成比較圖表 (增強版)"""
        if not self.results:
            print("⚠️  沒有結果資料，跳過圖表生成")
            return
        
        # 設定不顯示圖表
        plt.ioff()
        
        try:
            print("📊 開始生成比較圖表...")
            
            # 1. 模型比較圖（所有資料集的平均）
            self._plot_model_performance_summary()
            
            # 2. 各資料集詳細比較
            for dataset_name, results in self.results.items():
                self._plot_single_dataset_comparison(dataset_name, results)
            
            # 3. 模型穩定性分析
            self._plot_model_stability()
            
            # 4. 新增：預測散點圖
            self._plot_prediction_scatter()
            
            # 5. 新增：殘差分析圖
            self._plot_residual_analysis()
            
            print("✅ 所有圖表生成完成")
            
        except Exception as e:
            print(f"⚠️  圖表生成過程中發生錯誤: {e}")
            import traceback
            traceback.print_exc()
        finally:
            plt.ion()  # 重新開啟交互模式
    
    def _plot_prediction_scatter(self):
        """繪製預測vs實際值散點圖"""
        try:
            predictions_dir = os.path.join(config.RESULTS_DIR, "predictions")
            if not os.path.exists(predictions_dir):
                return
            
            csv_files = [f for f in os.listdir(predictions_dir) if f.endswith('_predictions.csv')]
            
            if not csv_files:
                return
            
            # 為每個模型創建一個總圖
            models = ['XGBoost', 'RandomForest', 'SVM']
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            colors = {'XGBoost': '#1f77b4', 'RandomForest': '#ff7f0e', 'SVM': '#2ca02c'}
            
            for i, model in enumerate(models):
                model_files = [f for f in csv_files if f.startswith(model)]
                
                all_actual = []
                all_predicted = []
                
                for csv_file in model_files:
                    try:
                        df = pd.read_csv(os.path.join(predictions_dir, csv_file))
                        all_actual.extend(df['actual'].tolist())
                        all_predicted.extend(df['predicted'].tolist())
                    except:
                        continue
                
                if all_actual and all_predicted:
                    axes[i].scatter(all_actual, all_predicted, 
                                  alpha=0.6, color=colors[model], s=10)
                    
                    # 完美預測線
                    min_val = min(min(all_actual), min(all_predicted))
                    max_val = max(max(all_actual), max(all_predicted))
                    axes[i].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)
                    
                    axes[i].set_xlabel('實際值', fontsize=12)
                    axes[i].set_ylabel('預測值', fontsize=12)
                    axes[i].set_title(f'{model} 預測散點圖', fontsize=14)
                    axes[i].grid(True, alpha=0.3)
                    
                    # 計算並顯示R²
                    from sklearn.metrics import r2_score
                    r2 = r2_score(all_actual, all_predicted)
                    axes[i].text(0.05, 0.95, f'R² = {r2:.3f}', 
                               transform=axes[i].transAxes, 
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            save_path = os.path.join(config.RESULTS_DIR, "plots", "prediction_scatter_all.png")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"📊 預測散點圖: {save_path}")
            
        except Exception as e:
            print(f"❌ 生成預測散點圖時發生錯誤: {e}")

    def _plot_residual_analysis(self):
        """繪製殘差分析圖"""
        try:
            predictions_dir = os.path.join(config.RESULTS_DIR, "predictions")
            if not os.path.exists(predictions_dir):
                return
            
            csv_files = [f for f in os.listdir(predictions_dir) if f.endswith('_predictions.csv')]
            
            if not csv_files:
                return
            
            models = ['XGBoost', 'RandomForest', 'SVM']
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            
            colors = {'XGBoost': '#1f77b4', 'RandomForest': '#ff7f0e', 'SVM': '#2ca02c'}
            
            for i, model in enumerate(models):
                model_files = [f for f in csv_files if f.startswith(model)]
                
                all_predicted = []
                all_residual = []
                
                for csv_file in model_files:
                    try:
                        df = pd.read_csv(os.path.join(predictions_dir, csv_file))
                        all_predicted.extend(df['predicted'].tolist())
                        all_residual.extend(df['residual'].tolist())
                    except:
                        continue
                
                if all_predicted and all_residual:
                    # 殘差vs預測值
                    axes[0, i].scatter(all_predicted, all_residual, 
                                     alpha=0.6, color=colors[model], s=10)
                    axes[0, i].axhline(y=0, color='r', linestyle='--', alpha=0.8)
                    axes[0, i].set_xlabel('預測值')
                    axes[0, i].set_ylabel('殘差')
                    axes[0, i].set_title(f'{model} 殘差圖')
                    axes[0, i].grid(True, alpha=0.3)
                    
                    # 殘差直方圖
                    axes[1, i].hist(all_residual, bins=50, alpha=0.7, color=colors[model])
                    axes[1, i].set_xlabel('殘差')
                    axes[1, i].set_ylabel('頻率')
                    axes[1, i].set_title(f'{model} 殘差分布')
                    axes[1, i].grid(True, alpha=0.3)
            
            plt.tight_layout()
            save_path = os.path.join(config.RESULTS_DIR, "plots", "residual_analysis.png")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"📊 殘差分析圖: {save_path}")
            
        except Exception as e:
            print(f"❌ 生成殘差分析圖時發生錯誤: {e}")
    
    def _plot_model_performance_summary(self):
        """繪製模型整體表現摘要"""
        try:
            # 收集所有結果
            all_results = []
            for dataset_name, results in self.results.items():
                for model_name, metrics in results.items():
                    all_results.append({
                        'model': model_name,
                        **metrics
                    })
            
            if not all_results:
                print("⚠️  沒有結果資料，跳過整體表現圖")
                return
            
            results_df = pd.DataFrame(all_results)
            model_avg = results_df.groupby('model')[config.COMPARISON_METRICS].mean()
            
            # 繪製圖表
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            axes = axes.ravel()
            
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
            
            for i, metric in enumerate(config.COMPARISON_METRICS):
                model_avg[metric].plot(kind='bar', ax=axes[i], color=colors[:len(model_avg)])
                axes[i].set_title(f'{metric} - 模型平均表現', fontsize=14)
                axes[i].set_ylabel(metric)
                axes[i].tick_params(axis='x', rotation=45)
                axes[i].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            save_path = os.path.join(config.RESULTS_DIR, "plots", "model_performance_summary.png")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"📊 模型整體表現圖: {save_path}")
            
        except Exception as e:
            print(f"❌ 生成整體表現圖時發生錯誤: {e}")
    
    def _plot_single_dataset_comparison(self, dataset_name, results):
        """繪製單一資料集的模型比較"""
        try:
            models = list(results.keys())
            metrics = config.COMPARISON_METRICS
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            axes = axes.ravel()
            
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
            
            for i, metric in enumerate(metrics):
                values = [results[model][metric] for model in models]
                bars = axes[i].bar(models, values, color=colors[:len(models)])
                axes[i].set_title(f'{metric} - {dataset_name[:30]}...', fontsize=12)
                axes[i].set_ylabel(metric)
                axes[i].tick_params(axis='x', rotation=45)
                axes[i].grid(True, alpha=0.3)
                
                # 在柱狀圖上標註數值
                for bar, value in zip(bars, values):
                    axes[i].text(bar.get_x() + bar.get_width()/2, 
                               bar.get_height() + bar.get_height()*0.01,
                               f'{value:.3f}', ha='center', va='bottom', fontsize=9)
            
            plt.tight_layout()
            
            # 清理檔案名稱
            clean_name = dataset_name.replace('/', '_').replace('\\', '_').replace(':', '_')
            clean_name = clean_name[:50]  # 限制長度
            save_path = os.path.join(config.RESULTS_DIR, "plots", f"comparison_{clean_name}.png")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"📊 {dataset_name[:30]}... 比較圖: {save_path}")
            
        except Exception as e:
            print(f"❌ 生成 {dataset_name} 比較圖時發生錯誤: {e}")
    
    def _plot_model_stability(self):
        """繪製模型穩定性分析"""
        try:
            all_results = []
            for dataset_name, results in self.results.items():
                for model_name, metrics in results.items():
                    all_results.append({
                        'model': model_name,
                        **metrics
                    })
            
            if not all_results:
                print("⚠️  沒有結果資料，跳過穩定性分析圖")
                return
            
            results_df = pd.DataFrame(all_results)
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            axes = axes.ravel()
            
            for i, metric in enumerate(config.COMPARISON_METRICS):
                if len(results_df) > 1:  # 只有在有多個結果時才畫盒鬚圖
                    results_df.boxplot(column=metric, by='model', ax=axes[i])
                    axes[i].set_title(f'{metric} - 模型穩定性分析')
                    axes[i].set_xlabel('模型')
                    axes[i].set_ylabel(metric)
                else:
                    axes[i].text(0.5, 0.5, '資料不足\n無法分析穩定性', 
                               ha='center', va='center', transform=axes[i].transAxes)
                    axes[i].set_title(f'{metric} - 穩定性分析')
            
            plt.tight_layout()
            
            save_path = os.path.join(config.RESULTS_DIR, "plots", "model_stability.png")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"📊 模型穩定性圖: {save_path}")
            
        except Exception as e:
            print(f"❌ 生成穩定性分析圖時發生錯誤: {e}")
    
    def generate_summary_report(self) -> Dict[str, Any]:
        """生成總結報告 (增強版)"""
        summary = {
            'timestamp': config.TIMESTAMP,
            'total_datasets': len(self.results),
            'total_models': len(list(self.results.values())[0]) if self.results else 0,
            'best_model_per_dataset': {},
            'overall_best': {},
            'model_performance_summary': {}
        }
        
        # 找出每個資料集的最佳模型（基於R²）
        for dataset_name, results in self.results.items():
            if results:
                best_model = max(results.items(), key=lambda x: x[1]['R2'])
                summary['best_model_per_dataset'][dataset_name] = {
                    'model': best_model[0],
                    'R2': best_model[1]['R2'],
                    'RMSE': best_model[1]['RMSE'],
                    'MAE': best_model[1]['MAE'],
                    'MAPE': best_model[1]['MAPE']
                }
        
        # 整體最佳組合
        best_combo = None
        best_r2 = -float('inf')
        
        for dataset_name, results in self.results.items():
            for model_name, metrics in results.items():
                if metrics['R2'] > best_r2:
                    best_r2 = metrics['R2']
                    best_combo = (dataset_name, model_name, metrics)
        
        if best_combo:
            summary['overall_best'] = {
                'dataset': best_combo[0],
                'model': best_combo[1],
                'R2': best_combo[2]['R2'],
                'RMSE': best_combo[2]['RMSE'],
                'MAE': best_combo[2]['MAE'],
                'MAPE': best_combo[2]['MAPE']
            }
        
        # 模型整體表現摘要
        all_results = []
        for dataset_name, results in self.results.items():
            for model_name, metrics in results.items():
                all_results.append({
                    'model': model_name,
                    **metrics
                })
        
        if all_results:
            results_df = pd.DataFrame(all_results)
            model_avg = results_df.groupby('model')[config.COMPARISON_METRICS].mean()
            summary['model_performance_summary'] = model_avg.to_dict()
        
        # 生成詳細分析
        try:
            detailed_analysis = self.generate_detailed_analysis()
            summary['detailed_analysis'] = detailed_analysis
        except Exception as e:
            print(f"⚠️  詳細分析生成失敗: {e}")
            summary['detailed_analysis'] = {}
        
        return summary

# 測試類別是否可以正確匯入
if __name__ == "__main__":
    print("✅ ModelEvaluator 類別測試")
    evaluator = ModelEvaluator()
    print("✅ ModelEvaluator 初始化成功")