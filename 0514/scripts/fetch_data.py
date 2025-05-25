# -*- coding: utf-8 -*-
import pandas as pd  # type: ignore
import numpy as np  # type: ignore
from ucimlrepo import fetch_ucirepo  # type: ignore
import os

# 建立儲存資料夾
os.makedirs("../data", exist_ok=True)

# ========== 載入資料集 ==========
wine_quality = fetch_ucirepo(id=186)
heart_disease = fetch_ucirepo(id=45)
breast_cancer = fetch_ucirepo(id=17)
online_retail = fetch_ucirepo(id=352)

# ========== Wine Quality ==========
X_wine = wine_quality.data.features.copy()
y_wine = wine_quality.data.targets

if isinstance(y_wine, pd.DataFrame):
    y_wine = y_wine.iloc[:, 0]
y_wine = (y_wine >= 7).astype(int)  # 品質 >= 7 為高品質（1）
y_wine.name = "target"

wine_data = pd.concat([X_wine, y_wine], axis=1)
wine_data.to_csv("../data/wine_processed.csv", index=False)

# ========== Heart Disease ==========
X_heart = heart_disease.data.features.copy()
y_heart = heart_disease.data.targets

if isinstance(y_heart, pd.DataFrame):
    y_heart = y_heart.iloc[:, 0]
if y_heart.nunique() > 2:
    y_heart = (y_heart > 0).astype(int)  # >0 為有病
y_heart.name = "target"

heart_data = pd.concat([X_heart, y_heart], axis=1)

# 清除非數值欄位、缺值
for col in heart_data.columns:
    if heart_data[col].dtype == object:
        heart_data[col] = pd.to_numeric(heart_data[col].replace('?', np.nan), errors='coerce')
heart_data = heart_data.dropna()

heart_data.to_csv("../data/heart_processed.csv", index=False)

# ========== Breast Cancer ==========
X_bc = breast_cancer.data.features.copy()
y_bc = breast_cancer.data.targets

if isinstance(y_bc, pd.DataFrame):
    y_bc = y_bc.iloc[:, 0]
y_bc = y_bc.map({'M': 1, 'B': 0})  # M=惡性, B=良性
y_bc.name = "target"

bc_data = pd.concat([X_bc, y_bc], axis=1)
bc_data.to_csv("../data/breast_cancer_processed.csv", index=False)

# ========== Online Retail ==========
X_or = online_retail.data.features.copy()
y_or = online_retail.data.targets
or_data = pd.concat([X_or, y_or], axis=1)

# 移除缺值與退貨資料（Invoice 以 C 開頭）
or_data = or_data.dropna()
invoice_col = [col for col in or_data.columns if 'invoice' in col.lower()]
if invoice_col:
    col_name = invoice_col[0]
    or_data = or_data[~or_data[col_name].astype(str).str.startswith('C')]

# 新增總金額欄位，標記大額訂單為 1
or_data['TotalAmount'] = or_data['Quantity'] * or_data['UnitPrice']
or_data['target'] = (or_data['TotalAmount'] > 100).astype(int)

# 保留欄位與 One-hot 編碼
or_data = or_data[['Quantity', 'UnitPrice', 'TotalAmount', 'Country', 'target']]
or_data = pd.get_dummies(or_data, columns=['Country'])

or_data.to_csv("../data/online_retail_processed.csv", index=False)

print("✅ 預處理完成，已儲存所有 processed CSV 檔案")


"""
| 結構          | 維度 | 像什麼      | 用途                 |
| ----------- | -- | -------- | ------------------ |
| `Series`    | 1D | 一欄數據     | 機器學習的 y（標籤、目標）     |
| `DataFrame` | 2D | 表格（多欄數據） | X 特徵資料、整份資料集的儲存與處理 |
"""

