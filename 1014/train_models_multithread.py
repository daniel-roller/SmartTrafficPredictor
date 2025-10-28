import os
import glob
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time
from config import Config

class MultiThreadModelTrainer:
    """多執行緒模型訓練器"""
    
    def __init__(self):
        self.config = Config()
        self.lock = threading.Lock()
        self.results = {}
        self.failed_jobs = []
    
    def build_lstm_model(self, input_shape, output_dim):
        """建立LSTM模型"""
        model = keras.Sequential([
            keras.layers.Input(shape=input_shape),
            keras.layers.LSTM(self.config.LSTM_PARAMS['units'], return_sequences=False),
            keras.layers.Dropout(self.config.LSTM_PARAMS['dropout']),
            keras.layers.Dense(output_dim)
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(self.config.LSTM_PARAMS['learning_rate']),
            loss='mse',
            metrics=['mae']
        )
        return model
    
    def build_tcn_model(self, input_shape, output_dim):
        """建立TCN模型 (簡化版)"""
        inputs = keras.layers.Input(shape=input_shape)
        
        # 簡單的Conv1D模擬TCN
        x = keras.layers.Conv1D(self.config.TCN_PARAMS['nb_filters'], 
                               self.config.TCN_PARAMS['kernel_size'], 
                               padding='causal', activation='relu')(inputs)
        x = keras.layers.Dropout(self.config.TCN_PARAMS['dropout_rate'])(x)
        x = keras.layers.Conv1D(self.config.TCN_PARAMS['nb_filters'], 
                               self.config.TCN_PARAMS['kernel_size'], 
                               padding='causal', activation='relu')(x)
        x = keras.layers.GlobalMaxPooling1D()(x)
        x = keras.layers.Dense(output_dim)(x)
        
        model = keras.Model(inputs, x)
        model.compile(
            optimizer=keras.optimizers.Adam(self.config.TCN_PARAMS['learning_rate']),
            loss='mse',
            metrics=['mae']
        )
        return model
    
    def build_transformer_model(self, input_shape, output_dim):
        """建立Transformer模型 (簡化版)"""
        inputs = keras.layers.Input(shape=input_shape)
        
        # 簡單的Multi-Head Attention
        attention_layer = keras.layers.MultiHeadAttention(
            num_heads=self.config.TRANSFORMER_PARAMS['num_heads'],
            key_dim=self.config.TRANSFORMER_PARAMS['model_dim'] // self.config.TRANSFORMER_PARAMS['num_heads']
        )
        
        x = attention_layer(inputs, inputs)
        x = keras.layers.Dropout(self.config.TRANSFORMER_PARAMS['dropout'])(x)
        x = keras.layers.GlobalAveragePooling1D()(x)
        x = keras.layers.Dense(self.config.TRANSFORMER_PARAMS['model_dim'], activation='relu')(x)
        x = keras.layers.Dropout(self.config.TRANSFORMER_PARAMS['dropout'])(x)
        x = keras.layers.Dense(output_dim)(x)
        
        model = keras.Model(inputs, x)
        model.compile(
            optimizer=keras.optimizers.Adam(self.config.TRANSFORMER_PARAMS['learning_rate']),
            loss='mse',
            metrics=['mae']
        )
        return model
    
    def train_single_model(self, dataset_path, model_type):
        """訓練單一模型"""
        try:
            # 載入資料集
            with open(dataset_path, 'rb') as f:
                data = pickle.load(f)
            
            X_train, y_train = data['X_train'], data['y_train']
            X_val, y_val = data['X_val'], data['y_val']
            X_test, y_test = data['X_test'], data['y_test']
            
            input_shape = (X_train.shape[1], X_train.shape[2])
            output_dim = y_train.shape[1]
            
            # 建立模型
            if model_type == 'LSTM':
                model = self.build_lstm_model(input_shape, output_dim)
                params = self.config.LSTM_PARAMS
            elif model_type == 'TCN':
                model = self.build_tcn_model(input_shape, output_dim)
                params = self.config.TCN_PARAMS
            elif model_type == 'Transformer':
                model = self.build_transformer_model(input_shape, output_dim)
                params = self.config.TRANSFORMER_PARAMS
            else:
                raise ValueError(f"未知的模型類型: {model_type}")
            
            # 設定回調
            dataset_name = os.path.basename(dataset_path).replace('.pkl', '')
            model_name = f"{model_type}_{dataset_name}"
            model_save_path = os.path.join(self.config.MODELS_DIR, f"{model_name}.h5")
            
            callbacks = [
                keras.callbacks.EarlyStopping(
                    patience=params['patience'], 
                    restore_best_weights=True,
                    monitor='val_loss'
                ),
                keras.callbacks.ModelCheckpoint(
                    model_save_path, 
                    save_best_only=True,
                    monitor='val_loss'
                ),
                keras.callbacks.ReduceLROnPlateau(
                    factor=0.5, 
                    patience=params['patience']//2,
                    monitor='val_loss'
                )
            ]
            
            # 訓練模型
            start_time = time.time()
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=params['epochs'],
                batch_size=params['batch_size'],
                callbacks=callbacks,
                verbose=0  # 減少輸出避免多執行緒混亂
            )
            train_time = time.time() - start_time
            
            # 評估模型
            train_loss, train_mae = model.evaluate(X_train, y_train, verbose=0)
            val_loss, val_mae = model.evaluate(X_val, y_val, verbose=0)
            test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
            
            # 計算額外指標
            y_pred = model.predict(X_test, verbose=0)
            
            # 反標準化計算實際誤差
            y_mean, y_std = data['y_mean'], data['y_std']
            y_test_real = y_test * y_std + y_mean
            y_pred_real = y_pred * y_std + y_mean
            
            rmse = np.sqrt(np.mean((y_test_real - y_pred_real) ** 2))
            mape = np.mean(np.abs((y_test_real - y_pred_real) / (y_test_real + 1e-8))) * 100
            
            result = {
                'model_type': model_type,
                'dataset_name': dataset_name,
                'segment_name': data['segment_name'],
                'window_size': data['window_size'],
                'horizon': data['horizon'],
                'train_time': train_time,
                'train_loss': train_loss,
                'train_mae': train_mae,
                'val_loss': val_loss,
                'val_mae': val_mae,
                'test_loss': test_loss,
                'test_mae': test_mae,
                'test_rmse': rmse,
                'test_mape': mape,
                'model_path': model_save_path,
                'epochs_trained': len(history.history['loss'])
            }
            
            with self.lock:
                key = f"{model_type}_{dataset_name}"
                self.results[key] = result
                print(f"✅ {key} 完成 (RMSE: {rmse:.2f}, MAPE: {mape:.2f}%, 時間: {train_time:.1f}s)")
            
            return result
                
        except Exception as e:
            error_key = f"{model_type}_{os.path.basename(dataset_path)}"
            with self.lock:
                self.failed_jobs.append((error_key, str(e)))
                print(f"❌ {error_key} 失敗: {e}")
            return None
    
    def train_all_models(self):
        """並行訓練所有模型"""
        dataset_files = glob.glob(os.path.join(self.config.DATASETS_DIR, "*.pkl"))
        model_types = ['LSTM', 'TCN', 'Transformer']
        
        print(f"🚀 開始並行訓練模型...")
        print(f"  📦 資料集數量: {len(dataset_files)}")
        print(f"  🤖 模型類型: {model_types}")
        print(f"  📊 總任務數: {len(dataset_files) * len(model_types)}")
        
        with ThreadPoolExecutor(max_workers=self.config.MAX_WORKERS_MODELS) as executor:
            future_to_job = {}
            
            # 提交所有訓練任務
            for dataset_path in dataset_files:
                for model_type in model_types:
                    future = executor.submit(self.train_single_model, dataset_path, model_type)
                    job_name = f"{model_type}_{os.path.basename(dataset_path)}"
                    future_to_job[future] = job_name
            
            # 收集結果
            completed = 0
            total_jobs = len(future_to_job)
            
            for future in as_completed(future_to_job):
                job_name = future_to_job[future]
                completed += 1
                
                try:
                    result = future.result()
                    if result:
                        progress = (completed / total_jobs) * 100
                        print(f"📈 進度: {progress:.1f}% ({completed}/{total_jobs})")
                except Exception as e:
                    with self.lock:
                        self.failed_jobs.append((job_name, str(e)))
                        print(f"❌ {job_name} 執行緒異常: {e}")
        
        # 儲存結果摘要
        self.save_training_summary()
        
        print(f"\n🎉 所有模型訓練完成!")
        print(f"  ✅ 成功: {len(self.results)}")
        print(f"  ❌ 失敗: {len(self.failed_jobs)}")
    
    def save_training_summary(self):
        """儲存訓練結果摘要"""
        if not self.results:
            return
        
        # 轉換為DataFrame
        results_df = pd.DataFrame(list(self.results.values()))
        
        # 儲存詳細結果
        results_path = os.path.join(self.config.RESULTS_DIR, "training_results.csv")
        results_df.to_csv(results_path, index=False)
        
        # 建立摘要統計
        summary_stats = results_df.groupby(['model_type', 'horizon']).agg({
            'test_rmse': ['mean', 'std', 'min', 'max'],
            'test_mape': ['mean', 'std', 'min', 'max'],
            'train_time': ['mean', 'sum']
        }).round(3)
        
        summary_path = os.path.join(self.config.RESULTS_DIR, "training_summary.csv")
        summary_stats.to_csv(summary_path)
        
        print(f"📊 結果已儲存:")
        print(f"  - 詳細結果: {results_path}")
        print(f"  - 摘要統計: {summary_path}")

def main():
    trainer = MultiThreadModelTrainer()
    trainer.train_all_models()

if __name__ == "__main__":
    main()