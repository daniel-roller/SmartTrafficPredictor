# 🚗 交通流量預測系統 - 使用指南

## 📋 系統簡介

這是一個基於傳統機器學習的交通流量預測系統，可以從 CSV 檔案載入數據並進行模型訓練與評估。

## 📂 專案結構

```
1014/
├── config.py              # 系統配置檔
├── data_loader.py         # CSV 數據載入器
├── data_processor.py      # 數據預處理與特徵工程
├── traditional_models.py  # 傳統機器學習模型 (Ridge, RandomForest, XGBoost)
├── evaluator.py          # 模型評估與可視化
├── main.py               # 主程式入口
├── utils.py              # 工具函數
├── select/               # CSV 原始檔案資料夾
│   ├── 國道1號_北向_台中系統交流道_后里地磅北.csv
│   ├── 國道1號_北向_高科交流道_路竹交流道.csv
│   ├── 國道1號_南向_后里地磅南_台中系統交流道.csv
│   └── 國道1號_南向_路竹交流道_高科交流道.csv
├── results/              # 實驗結果輸出資料夾
└── README.md             # 本檔案
```

## 🚀 快速開始

### 1. 安裝依賴套件

```bash
pip install numpy pandas scikit-learn xgboost matplotlib seaborn
```

### 2. 準備數據

將交通流量 CSV 檔案放入 `select/` 資料夾。CSV 檔案應包含以下欄位：
- `Speed`: 車速
- `Flow`: 流量
- `dataTime`: 時間戳記
- 其他輔助欄位 (VD1, VD2, VD3, 補值等)

### 3. 執行完整流程

```bash
cd 1014
python main.py
```

系統會自動：
1. 從 CSV 載入數據
2. 進行特徵工程
3. 訓練 3 種模型 (Ridge, RandomForest, XGBoost)
4. 評估模型性能
5. 生成視覺化報告

### 4. 單獨測試數據載入

```bash
python data_loader.py
```

## 📊 輸出結果

執行完成後，在 `results/experiment_YYYYMMDD_HHMMSS/` 資料夾中會生成：

### 主要檔案
- **整體模型比較儀表板.png** - 所有模型的視覺化比較
- **模型比較摘要報告.md** - Markdown 格式的摘要報告
- **模型比較結果.csv** - 詳細的數值結果

### 子資料夾
- `models/` - 訓練好的模型檔案 (.pkl)
- `plots/` - 視覺化圖表
- `logs/` - 執行日誌

## ⚙️ 系統配置

可以在 `config.py` 中調整以下參數：

### 數據處理參數
```python
MAX_SAMPLE_SIZE = 10000    # 每個資料集的最大樣本數
TRAIN_RATIO = 0.7          # 訓練集比例
VAL_RATIO = 0.15           # 驗證集比例
TEST_RATIO = 0.15          # 測試集比例
SCALER_METHOD = 'standard' # 標準化方法
```

### 特徵工程參數
```python
LAG_FEATURES = [1, 2, 3]   # 滯後特徵窗口
```

### 模型參數
```python
TRADITIONAL_ML_PARAMS = {
    'Ridge': {...},
    'RandomForest': {...},
    'XGBoost': {...}
}
```

## 🤖 支援的模型

系統預設使用 3 種傳統機器學習模型：

1. **Ridge Regression** - 線性迴歸模型
   - 優點：訓練快速、解釋性強
   - 適用於：線性關係明顯的數據

2. **Random Forest** - 隨機森林
   - 優點：準確度高、魯棒性強
   - 適用於：複雜的非線性關係

3. **XGBoost** - 梯度提升樹
   - 優點：性能卓越、處理速度快
   - 適用於：大規模數據預測

## 📈 評估指標

系統使用以下指標評估模型性能：

- **R²** - 決定係數（越接近 1 越好）
- **RMSE** - 均方根誤差（越小越好）
- **MAE** - 平均絕對誤差（越小越好）
- **MAPE** - 平均絕對百分比誤差（越小越好）
- **訓練時間** - 模型訓練所需時間

## 🔧 進階使用

### 自訂時間窗口

在 `data_loader.py` 中可以調整時間窗口大小：

```python
datasets = loader.load_all_csv_datasets(
    window_size=12,        # 使用過去 12 個時間點
    prediction_horizon=1   # 預測未來 1 個時間點
)
```

### 儲存為 .npy 格式

如果想要加速後續實驗，可以將 CSV 轉換為 .npy 格式：

```python
python data_loader.py
# 當提示時輸入 'y' 來儲存
```

### 只訓練特定模型

修改 `traditional_models.py` 中的 `create_models()` 方法，註解掉不需要的模型。

## 📝 實驗記錄

每次實驗會自動生成唯一的實驗 ID（基於時間戳記），所有結果都會保存在對應的資料夾中，方便追蹤和比較不同實驗。

## ⚠️ 注意事項

1. **記憶體使用**：處理大型 CSV 檔案時，系統會自動限制樣本數量到 `MAX_SAMPLE_SIZE`
2. **計算時間**：隨機森林模型訓練時間較長（約 2 秒/資料集）
3. **中文顯示**：系統會自動嘗試設定中文字體，如果顯示異常請檢查系統字體

## 🐛 常見問題

### Q: 執行時出現 "找不到資料來源" 錯誤？
A: 請確認 `select/` 資料夾存在且包含 CSV 檔案。

### Q: 模型訓練時記憶體不足？
A: 減少 `config.py` 中的 `MAX_SAMPLE_SIZE` 參數。

### Q: 圖表中文顯示為方塊？
A: 系統使用的是備用字體，結果仍然可用，只是顯示效果受影響。

### Q: 想要調整模型參數？
A: 修改 `config.py` 中的 `TRADITIONAL_ML_PARAMS` 字典。

## 📧 聯絡資訊

如有問題或建議，請聯繫專案維護者。

---

**版本**: 1.0  
**最後更新**: 2025-10-24  
**授權**: MIT License
