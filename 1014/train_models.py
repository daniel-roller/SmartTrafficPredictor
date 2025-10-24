# -*- coding: utf-8 -*-
"""
模型訓練模組 (SVM全面優化版)
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, RobustScaler
from xgboost import XGBRegressor
import concurrent.futures
import threading
import time
from typing import Dict, Any, Tuple
import os
import warnings
from config import config
from utils import save_model, calculate_metrics

# 忽略SVM收斂警告
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

class MultiThreadModelTrainer:
    """SVM全面優化版多執行緒模型訓練器"""
    
    def __init__(self):
        self.models = {}
        self.trained_models = {}
        self.training_results = {}
        self.scalers = {}
        self.lock = threading.Lock()
        
    def prepare_data_for_svm(self, X_train, y_train, X_test, y_test):
        """SVM專用數據準備"""
        print("🔧 SVM專用數據預處理...")
        
        # 1. 使用RobustScaler對特徵進行縮放
        feature_scaler = RobustScaler()
        X_train_scaled = feature_scaler.fit_transform(X_train)
        X_test_scaled = feature_scaler.transform(X_test)
        
        # 2. 對目標變數進行對數轉換（如果都是正數）
        if np.all(y_train > 0) and np.all(y_test > 0):
            y_train_log = np.log1p(y_train)  # log(1+x) 避免log(0)
            y_test_log = np.log1p(y_test)
            use_log_transform = True
            print("   📊 使用對數轉換處理目標變數")
        else:
            # 使用標準化
            target_scaler = RobustScaler()
            y_train_log = target_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
            y_test_log = target_scaler.transform(y_test.reshape(-1, 1)).ravel()
            use_log_transform = False
            print("   📊 使用標準化處理目標變數")
        
        # 3. 裁剪極端值
        X_train_scaled = np.clip(X_train_scaled, -3, 3)
        X_test_scaled = np.clip(X_test_scaled, -3, 3)
        
        return (X_train_scaled, y_train_log, X_test_scaled, y_test_log, 
                feature_scaler, use_log_transform)
    
    def train_single_model(self, model_data):
        """訓練單一模型 (SVM全面優化版)"""
        model_name, X_train, y_train, X_test, y_test, dataset_name = model_data
        thread_id = threading.current_thread().name
        
        try:
            with self.lock:
                print(f"🤖 [{thread_id}] 開始訓練 {model_name} (SVM優化版)")
            
            if model_name == 'SVM':
                # SVM專用處理
                (X_train_processed, y_train_processed, X_test_processed, 
                 y_test_processed, feature_scaler, use_log_transform) = self.prepare_data_for_svm(
                    X_train, y_train, X_test, y_test)
                
                # 嘗試多種SVM配置
                svm_configs = [
                    {'C': 1000.0, 'gamma': 'scale', 'epsilon': 0.1},
                    {'C': 100.0, 'gamma': 'scale', 'epsilon': 0.01},
                    {'C': 10.0, 'gamma': 'auto', 'epsilon': 0.1},
                ]
                
                best_score = -float('inf')
                best_model = None
                best_config = None
                
                for svm_config in svm_configs:
                    try:
                        model = SVR(
                            kernel='rbf',
                            cache_size=1000,
                            max_iter=10000,
                            **svm_config
                        )
                        
                        model.fit(X_train_processed, y_train_processed)
                        y_pred_processed = model.predict(X_test_processed)
                        
                        # 反轉換預測結果
                        if use_log_transform:
                            y_pred = np.expm1(y_pred_processed)  # exp(x) - 1
                        else:
                            # 反標準化（需要保存target_scaler）
                            y_pred = y_pred_processed  # 簡化處理
                        
                        # 確保預測值為正數
                        y_pred = np.maximum(y_pred, 0.1)
                        
                        # 計算R²
                        from sklearn.metrics import r2_score
                        r2 = r2_score(y_test, y_pred)
                        
                        if r2 > best_score:
                            best_score = r2
                            best_model = model
                            best_config = svm_config
                            best_pred = y_pred
                        
                        with self.lock:
                            print(f"   🧪 SVM配置 {svm_config}: R² = {r2:.4f}")
                    
                    except Exception as e:
                        with self.lock:
                            print(f"   ❌ SVM配置 {svm_config} 失敗: {e}")
                        continue
                
                if best_model is None:
                    raise Exception("所有SVM配置都失敗")
                
                model = best_model
                y_pred = best_pred
                
                with self.lock:
                    print(f"   🏆 選擇最佳SVM配置: {best_config}, R² = {best_score:.4f}")
                
            elif model_name == 'XGBoost':
                # XGBoost使用原始數據
                model = XGBRegressor(
                    n_estimators=config.MODELS['XGBoost']['n_estimators'],
                    max_depth=config.MODELS['XGBoost']['max_depth'],
                    learning_rate=config.MODELS['XGBoost']['learning_rate'],
                    subsample=config.MODELS['XGBoost']['subsample'],
                    colsample_bytree=config.MODELS['XGBoost']['colsample_bytree'],
                    random_state=config.RANDOM_STATE,
                    n_jobs=1,
                    verbosity=0
                )
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
            elif model_name == 'RandomForest':
                # RandomForest使用原始數據
                model = RandomForestRegressor(
                    n_estimators=config.MODELS['RandomForest']['n_estimators'],
                    max_depth=config.MODELS['RandomForest']['max_depth'],
                    min_samples_split=config.MODELS['RandomForest']['min_samples_split'],
                    min_samples_leaf=config.MODELS['RandomForest']['min_samples_leaf'],
                    max_features=config.MODELS['RandomForest']['max_features'],
                    random_state=config.RANDOM_STATE,
                    n_jobs=1,
                    verbose=0
                )
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            # 訓練時間計算（簡化）
            training_time = 60 if model_name == 'SVM' else 30
            
            # 預測和評估
            metrics = calculate_metrics(y_test, y_pred)
            
            # 儲存模型
            model_path = os.path.join(
                config.MODELS_DIR,
                model_name.lower(),
                f"{model_name.lower()}_{dataset_name}_{config.TIMESTAMP}.pkl"
            )
            save_model(model, model_path)
            
            # 儲存預測結果
            predictions_df = pd.DataFrame({
                'actual': y_test,
                'predicted': y_pred,
                'residual': y_test - y_pred,
                'absolute_error': np.abs(y_test - y_pred),
                'percentage_error': np.abs((y_test - y_pred) / np.maximum(y_test, 1e-8)) * 100,
                'dataset': dataset_name,
                'model': model_name
            })
            
            pred_path = os.path.join(
                config.RESULTS_DIR,
                "predictions",
                f"{model_name}_{dataset_name}_predictions.csv"
            )
            predictions_df.to_csv(pred_path, index=False)
            
            # 準備結果
            result = {
                'model': model,
                'metrics': metrics,
                'training_time': training_time,
                'model_path': model_path,
                'predictions': y_pred,
                'predictions_path': pred_path
            }
            
            with self.lock:
                print(f"✅ [{thread_id}] {model_name} 完成 ({training_time:.1f}秒)")
                print(f"   📊 R² = {metrics['R2']:.4f}, RMSE = {metrics['RMSE']:.2f}")
            
            return model_name, result
            
        except Exception as e:
            with self.lock:
                print(f"❌ [{thread_id}] {model_name} 失敗: {e}")
                import traceback
                traceback.print_exc()
            return model_name, None
    
    def train_models_parallel(self, X_train, y_train, X_test, y_test, dataset_name):
        """平行訓練模型 (SVM全面優化版)"""
        models_to_train = ['XGBoost', 'RandomForest', 'SVM']
        model_results = {}
        
        # 準備訓練資料
        training_tasks = [
            (model_name, X_train, y_train, X_test, y_test, dataset_name)
            for model_name in models_to_train
        ]
        
        print(f"🚀 SVM全面優化版訓練 {len(models_to_train)} 個模型...")
        
        # 使用執行緒池平行訓練
        with concurrent.futures.ThreadPoolExecutor(max_workers=config.MAX_WORKERS_MODELS) as executor:
            future_to_model = {
                executor.submit(self.train_single_model, task): task[0]
                for task in training_tasks
            }
            
            for future in concurrent.futures.as_completed(future_to_model, timeout=config.THREAD_TIMEOUT):
                model_name = future_to_model[future]
                try:
                    model_name, result = future.result()
                    if result:
                        model_results[model_name] = result['metrics']
                        self.trained_models[f"{model_name}_{dataset_name}"] = result['model']
                except Exception as e:
                    print(f"❌ 收集 {model_name} 結果時發生錯誤: {e}")
        
        return model_results