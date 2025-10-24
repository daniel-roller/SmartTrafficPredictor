# -*- coding: utf-8 -*-
"""
交通流量預測系統 - 傳統機器學習模型 (簡化版)
"""

import numpy as np
import pandas as pd
import warnings
from typing import Dict, List, Tuple, Any, Optional
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import joblib
import os

from config import config
from utils import (
    save_model, Timer, print_subsection_header, 
    print_progress_bar, format_number
)

warnings.filterwarnings('ignore')

class TraditionalMLModels:
    """傳統機器學習模型訓練器 (簡化版)"""
    
    def __init__(self):
        self.models = {}
        self.results = {}
        self.timer = Timer()
        
    def create_models(self) -> Dict:
        """建立模型實例 (只保留3個)"""
        models = {}
        
        # 線性模型
        models['Ridge'] = Ridge(**config.TRADITIONAL_ML_PARAMS['Ridge'])
        
        # 樹模型
        models['RandomForest'] = RandomForestRegressor(**config.TRADITIONAL_ML_PARAMS['RandomForest'])
        
        # 梯度提升模型
        models['XGBoost'] = xgb.XGBRegressor(**config.TRADITIONAL_ML_PARAMS['XGBoost'])
        
        return models
    
    def train_single_model(self, model_name: str, model, 
                          X_train: np.ndarray, y_train: np.ndarray,
                          X_val: np.ndarray = None, y_val: np.ndarray = None) -> Dict:
        """訓練單一模型"""
        print(f"  🔄 訓練 {model_name}...")
        self.timer.start()
        
        try:
            # 訓練模型
            if model_name == 'XGBoost' and X_val is not None:
                # XGBoost使用驗證集
                model.fit(
                    X_train, y_train.ravel(),
                    eval_set=[(X_val, y_val.ravel())],
                    verbose=False
                )
            else:
                model.fit(X_train, y_train.ravel())
            
            self.timer.stop()
            
            result = {
                'model_name': model_name,
                'model': model,
                'training_time': self.timer.elapsed(),
                'status': 'success'
            }
            
            print(f"    ✅ {model_name} 訓練完成 ({self.timer.elapsed_str()})")
            return result
            
        except Exception as e:
            self.timer.stop()
            print(f"    ❌ {model_name} 訓練失敗: {e}")
            return {
                'model_name': model_name,
                'model': None,
                'training_time': 0,
                'status': 'failed',
                'error': str(e)
            }
    
    def predict_and_evaluate(self, model: Any, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """預測並評估模型"""
        try:
            # 預測
            y_pred: np.ndarray = model.predict(X_test)
            y_true: np.ndarray = y_test.ravel()
            y_pred = y_pred.ravel()
            
            # 計算評估指標
            r2 = r2_score(y_true, y_pred)
            mse = mean_squared_error(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            
            # MAPE (避免除零)
            mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
            
            # 方向準確度
            if len(y_true) > 1:
                true_direction = np.diff(y_true) > 0
                pred_direction = np.diff(y_pred) > 0
                direction_accuracy = np.mean(true_direction == pred_direction) * 100
            else:
                direction_accuracy = 0.0
            
            metrics = {
                'R²': r2,
                'MSE': mse,
                'MAE': mae,
                'RMSE': rmse,
                'MAPE': mape,
                'Direction_Accuracy': direction_accuracy,
                'predictions': y_pred,
                'true_values': y_true
            }
            
            return metrics
            
        except Exception as e:
            print(f"    ❌ 評估失敗: {e}")
            return {
                'R²': 0, 'MSE': float('inf'), 'MAE': float('inf'),
                'RMSE': float('inf'), 'MAPE': float('inf'), 
                'Direction_Accuracy': 0, 'error': str(e)
            }
    
    def cross_validate_model(self, model: Any, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, float]:
        """交叉驗證"""
        try:
            tscv = TimeSeriesSplit(n_splits=config.CV_FOLDS)
            
            # R²分數
            cv_r2 = cross_val_score(model, X_train, y_train.ravel(), 
                                   cv=tscv, scoring='r2', n_jobs=-1)
            
            # MSE分數
            cv_mse = -cross_val_score(model, X_train, y_train.ravel(), 
                                     cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1)
            
            cv_results = {
                'cv_r2_mean': np.mean(cv_r2),
                'cv_r2_std': np.std(cv_r2),
                'cv_mse_mean': np.mean(cv_mse),
                'cv_mse_std': np.std(cv_mse)
            }
            
            return cv_results
            
        except Exception as e:
            print(f"    ⚠️ 交叉驗證失敗: {e}")
            return {
                'cv_r2_mean': 0.0, 'cv_r2_std': 0.0,
                'cv_mse_mean': float('inf'), 'cv_mse_std': 0.0
            }
    
    def train_all_models(self, processed_data: Dict) -> Dict:
        """訓練所有模型"""
        dataset_name = processed_data['name']
        print_subsection_header(f"🤖 訓練傳統ML模型 - {dataset_name}")
        
        X_train = processed_data['X_train']
        X_val = processed_data['X_val']
        X_test = processed_data['X_test']
        y_train = processed_data['y_train']
        y_val = processed_data['y_val']
        y_test = processed_data['y_test']
        
        print(f"📊 資料概況:")
        print(f"  - 特徵數: {X_train.shape[1]}")
        print(f"  - 訓練樣本: {len(X_train)}")
        print(f"  - 驗證樣本: {len(X_val)}")
        print(f"  - 測試樣本: {len(X_test)}")
        
        # 建立模型
        models = self.create_models()
        results = {}
        
        total_models = len(models)
        
        for i, (model_name, model) in enumerate(models.items()):
            print_progress_bar(i, total_models, f"訓練模型", f"{model_name}")
            
            # 訓練模型
            train_result = self.train_single_model(
                model_name, model, X_train, y_train, X_val, y_val
            )
            
            if train_result['status'] == 'success':
                trained_model = train_result['model']
                
                # 驗證集評估
                val_metrics = self.predict_and_evaluate(trained_model, X_val, y_val)
                
                # 測試集評估
                test_metrics = self.predict_and_evaluate(trained_model, X_test, y_test)
                
                # 交叉驗證
                cv_results = self.cross_validate_model(trained_model, X_train, y_train)
                
                # 儲存模型
                if config.SAVE_MODELS:
                    model_path = os.path.join(
                        config.MODELS_DIR, 
                        f"{dataset_name}_{model_name}_model.pkl"
                    )
                    save_model(trained_model, model_path, model_name)
                
                # 整合結果
                results[model_name] = {
                    'dataset': dataset_name,
                    'model_name': model_name,
                    'model': trained_model,
                    'training_time': train_result['training_time'],
                    'val_metrics': val_metrics,
                    'test_metrics': test_metrics,
                    'cv_results': cv_results,
                    'feature_count': X_train.shape[1]
                }
                
                # 顯示驗證結果
                print(f"    📈 驗證 R²: {val_metrics['R²']:.4f}, RMSE: {val_metrics['RMSE']:.4f}")
                
            else:
                results[model_name] = train_result
        
        print_progress_bar(total_models, total_models, "訓練完成", "")
        
        # 找出最佳模型
        valid_models = [name for name in results.keys() if results[name].get('val_metrics')]
        if valid_models:
            best_model_name = max(valid_models, key=lambda x: results[x]['val_metrics']['R²'])
            best_r2 = results[best_model_name]['val_metrics']['R²']
            print(f"\n🏆 最佳模型: {best_model_name} (R² = {best_r2:.4f})")
        
        return results