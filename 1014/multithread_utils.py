# -*- coding: utf-8 -*-
"""
多執行緒工具函數
"""

import threading
import time
import psutil
import os
from typing import Dict, List, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

class ThreadSafeCounter:
    """執行緒安全計數器"""
    def __init__(self, initial_value=0):
        self.value = initial_value
        self.lock = threading.Lock()
    
    def increment(self):
        with self.lock:
            self.value += 1
            return self.value
    
    def get_value(self):
        with self.lock:
            return self.value

class ProgressTracker:
    """進度追蹤器"""
    def __init__(self, total_tasks):
        self.total_tasks = total_tasks
        self.completed = ThreadSafeCounter()
        self.start_time = time.time()
        self.lock = threading.Lock()
    
    def update_progress(self, task_name=""):
        completed = self.completed.increment()
        elapsed_time = time.time() - self.start_time
        
        if self.total_tasks > 0:
            progress = completed / self.total_tasks * 100
            avg_time = elapsed_time / completed if completed > 0 else 0
            estimated_remaining = (self.total_tasks - completed) * avg_time
            
            with self.lock:
                print(f"📈 進度: {completed}/{self.total_tasks} ({progress:.1f}%) "
                      f"- 預估剩餘: {estimated_remaining:.1f}秒 {task_name}")

class ResourceMonitor:
    """系統資源監控器"""
    def __init__(self):
        self.monitoring = False
        self.monitor_thread = None
        self.lock = threading.Lock()
        
    def start_monitoring(self, interval=5):
        """開始監控系統資源"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, args=(interval,))
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        print("📊 系統資源監控已啟動")
    
    def stop_monitoring(self):
        """停止監控"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        print("📊 系統資源監控已停止")
    
    def _monitor_loop(self, interval):
        """監控迴圈"""
        while self.monitoring:
            try:
                cpu_percent = psutil.cpu_percent(interval=1)
                memory_percent = psutil.virtual_memory().percent
                
                with self.lock:
                    if cpu_percent > 90 or memory_percent > 85:
                        print(f"⚠️  系統資源警告: CPU {cpu_percent:.1f}%, 記憶體 {memory_percent:.1f}%")
                
                time.sleep(interval)
            except Exception as e:
                print(f"❌ 監控錯誤: {e}")
                break

def parallel_execute(tasks, max_workers=None, task_name="Task"):
    """執行平行任務的通用函數"""
    if not tasks:
        return []
    
    results = []
    progress = ProgressTracker(len(tasks))
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任務
        future_to_task = {executor.submit(task): f"{task_name}_{i}" 
                         for i, task in enumerate(tasks)}
        
        # 收集結果
        for future in as_completed(future_to_task):
            task_id = future_to_task[future]
            try:
                result = future.result()
                results.append(result)
                progress.update_progress(task_id)
            except Exception as e:
                print(f"❌ {task_id} 執行失敗: {e}")
                results.append(None)
    
    return results

def safe_thread_operation(operation, *args, **kwargs):
    """執行緒安全操作裝飾器"""
    def wrapper():
        try:
            return operation(*args, **kwargs)
        except Exception as e:
            thread_id = threading.current_thread().name
            print(f"❌ [{thread_id}] 操作失敗: {e}")
            return None
    return wrapper