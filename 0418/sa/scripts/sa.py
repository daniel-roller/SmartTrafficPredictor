# segmentation_analysis.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns # 常用於資料視覺化

from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.metrics import (classification_report, confusion_matrix, ConfusionMatrixDisplay,
                             mean_squared_error, r2_score)

# 設定亂數種子以便結果重現
RANDOM_STATE = 42
N_SAMPLES = 1000

# --- 1. 資料模擬與前處理 ---
def simulate_customer_data(n_samples=N_SAMPLES, random_state=RANDOM_STATE):
    """
    模擬顧客資料，包含分類、回歸和分群所需的特徵與目標變數。
    """
    print("--- 正在模擬顧客資料 ---")

    # 模擬分類資料：產生與 Segment 相關的特徵和目標
    # 使用 4 個特徵，其中 3 個有訊息量，1 個多餘
    X_clf, y_clf = make_classification(
        n_samples=n_samples,
        n_features=4,
        n_informative=3,
        n_redundant=1,
        n_classes=3, # 模擬 3 個 Segment
        n_clusters_per_class=1, # 讓每個 Segment 內部較緊密
        flip_y=0.01, # 加入少量標籤噪音
        random_state=random_state
    )

    # 模擬回歸資料：產生與 PurchaseAmount 相關的特徵和目標
    # 使用與分類相同的特徵結構，但目標是連續值
    X_reg, y_reg = make_regression(
        n_samples=n_samples,
        n_features=4, # 使用相同數量的特徵
        n_informative=4, # 假設所有特徵都與購買金額有關
        noise=10.0, # 加入噪音
        random_state=random_state
    )

    # 將模擬的特徵和目標整合成 DataFrame
    # 假設 make_ functions 產生的特徵順序對應我們的欄位名稱
    df = pd.DataFrame(X_clf, columns=['Age', 'Income', 'PurchaseFreq', 'Membership'])

    # 將數值特徵縮放到更像真實世界的範圍 (簡單線性縮放示意)
    # 注意：這只是示意，真實資料可能需要更複雜的處理
    df['Age'] = (df['Age'] * 5 + 30).astype(int) # 年齡約 20-80 歲
    df['Income'] = (df['Income'] * 5000 + 50000).astype(int) # 收入約 3-10 萬
    df['PurchaseFreq'] = (df['PurchaseFreq'] * 2 + 5).astype(int).clip(lower=1, upper=20) # 購買頻率約 1-20 次
    # Membership 本身就是 binary，假設 make_classification 產生的是 -1/1 或 0/1，我們轉換為 0/1
    df['Membership'] = (df['Membership'] > 0).astype(int)

    # 加入模擬的 Segment 欄位 (來自分類目標)
    segment_map = {0: 'Segment A', 1: 'Segment B', 2: 'Segment C'}
    df['Segment'] = y_clf
    df['Segment'] = df['Segment'].map(segment_map)

    # 加入模擬的 PurchaseAmount 欄位 (來自回歸目標)
    # 確保 PurchaseAmount 是正值
    df['PurchaseAmount'] = (y_reg + abs(y_reg.min()) + 50).astype(int) # 確保最小值為正，並加上偏移量
    df['PurchaseAmount'] = df['PurchaseAmount'].clip(lower=50) # 最低購買金額

    # 加入隨機產生的 Gender 欄位 (不用於模型，僅供參考)
    df['Gender'] = np.random.choice(['Male', 'Female', 'Other'], size=n_samples)

    print("資料模擬完成。資料概覽：")
    print(df.head())
    print("\n資料資訊：")
    df.info()
    print("\n資料描述統計：")
    print(df.describe())

    return df

# --- 2. 分群分析 (Clustering) ---
def perform_clustering(df, features, n_clusters, random_state=RANDOM_STATE):
    """
    使用 KMeans 進行分群並透過 PCA 降維後視覺化。
    """
    print(f"\n--- 正在執行 KMeans 分群 (k={n_clusters}) ---")

    X_cluster = df[features]

    # 標準化特徵，KMeans 對尺度敏感
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)

    # 使用 PCA 將資料降至 2 維以便視覺化
    pca = PCA(n_components=2, random_state=random_state)
    X_pca = pca.fit_transform(X_scaled)

    # 執行 KMeans 分群
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10) # n_init=10 建議值
    cluster_labels = kmeans.fit_predict(X_scaled) # 注意：KMeans 訓練使用標準化後的資料

    df[f'KMeans_Cluster_k{n_clusters}'] = cluster_labels

    print(f"KMeans 分群完成 (k={n_clusters})。各群組樣本數：")
    print(df[f'KMeans_Cluster_k{n_clusters}'].value_counts().sort_index())

    # 視覺化分群結果
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis', s=50, alpha=0.7)
    plt.title(f'KMeans Clustering (k={n_clusters}) on PCA Components')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.colorbar(scatter, label='Cluster ID')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

    # 簡單分析各群組特徵均值 (使用原始資料的均值)
    print(f"\n各 KMeans 群組 (k={n_clusters}) 在原始特徵上的平均值：")
    print(df.groupby(f'KMeans_Cluster_k{n_clusters}')[features].mean())


# --- 3. 分類任務 (Classification) ---
def perform_classification(df, features, target_col, random_state=RANDOM_STATE):
    """
    使用 RandomForestClassifier 預測顧客 Segment。
    """
    print("\n--- 正在執行分類任務 (預測 Segment) ---")

    X = df[features]
    y = df[target_col]

    # 分割訓練集和測試集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=random_state, stratify=y # stratify=y 確保訓練測試集類別比例一致
    )

    print(f"訓練集樣本數: {len(X_train)}")
    print(f"測試集樣本數: {len(X_test)}")

    # 標準化特徵 (對於基於樹的模型非必須，但對其他模型如 SVM, LogReg 有幫助)
    # 在這裡為了演示可以選擇性使用，但對於 RandomForest 影響不大
    # scaler_clf = StandardScaler()
    # X_train_scaled = scaler_clf.fit_transform(X_train)
    # X_test_scaled = scaler_clf.transform(X_test)

    # 建立 RandomForestClassifier 模型
    model = RandomForestClassifier(n_estimators=100, random_state=random_state, class_weight='balanced') # class_weight='balanced' 處理可能的類別不平衡

    # 訓練模型
    print("正在訓練 RandomForest 分類模型...")
    model.fit(X_train, y_train) # 使用原始或標準化資料皆可，取決於上面是否標準化

    # 在測試集上進行預測
    y_pred = model.predict(X_test) # 使用原始或標準化測試資料

    # 模型評估
    print("\n分類報告：")
    print(classification_report(y_test, y_pred))

    # 混淆矩陣視覺化
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)

    print("\n混淆矩陣：")
    plt.figure(figsize=(8, 6))
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix for Segment Classification')
    plt.grid(False) # 關閉網格線，避免影響矩陣圖顯示
    plt.show()

    # 輸出重要特徵 (僅 RandomForest 有此功能)
    print("\n特徵重要性 (Feature Importance)：")
    feature_importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    print(feature_importances)
    plt.figure(figsize=(8, 5))
    feature_importances.plot(kind='bar')
    plt.title('Feature Importance for Segment Classification')
    plt.ylabel('Importance')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout() # 調整佈局，避免標籤重疊
    plt.show()


# --- 4. 回歸任務 (Regression) ---
def perform_regression(df, features, target_col, random_state=RANDOM_STATE):
    """
    使用 GradientBoostingRegressor 預測顧客 PurchaseAmount。
    """
    print("\n--- 正在執行回歸任務 (預測 PurchaseAmount) ---")

    X = df[features]
    y = df[target_col]

    # 分割訓練集和測試集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=random_state
    )

    print(f"訓練集樣本數: {len(X_train)}")
    print(f"測試集樣本數: {len(X_test)}")

    # 標準化特徵 (同樣，對於基於樹的模型非必須)
    # scaler_reg = StandardScaler()
    # X_train_scaled = scaler_reg.fit_transform(X_train)
    # X_test_scaled = scaler_reg.transform(X_test)

    # 建立 GradientBoostingRegressor 模型
    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=random_state)

    # 訓練模型
    print("正在訓練 Gradient Boosting 回歸模型...")
    model.fit(X_train, y_train) # 使用原始或標準化資料皆可

    # 在測試集上進行預測
    y_pred = model.predict(X_test) # 使用原始或標準化測試資料

    # 模型評估
    rmse = mean_squared_error(y_test, y_pred, squared=False) # squared=False 返回 RMSE
    r2 = r2_score(y_test, y_pred)

    print(f"\n回歸模型評估：")
    print(f"RMSE (均方根誤差): {rmse:.2f}")
    print(f"R² (決定係數): {r2:.2f}")

    # 預測值 vs 實際值散點圖
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label='理想預測線') # 繪製理想預測線
    plt.xlabel('實際購買金額')
    plt.ylabel('預測購買金額')
    plt.title('實際 vs 預測購買金額')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

    # 輸出重要特徵
    print("\n特徵重要性 (Feature Importance)：")
    feature_importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    print(feature_importances)
    plt.figure(figsize=(8, 5))
    feature_importances.plot(kind='bar')
    plt.title('Feature Importance for Purchase Amount Regression')
    plt.ylabel('Importance')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout() # 調整佈局
    plt.show()


# --- 主程式執行區塊 ---
if __name__ == "__main__":
    # 模擬資料
    customer_df = simulate_customer_data()

    # 定義各任務使用的特徵
    clustering_features = ['Age', 'Income', 'PurchaseFreq']
    classification_regression_features = ['Age', 'Income', 'PurchaseFreq', 'Membership']
    classification_target = 'Segment'
    regression_target = 'PurchaseAmount'

    # 執行分群分析並探索不同 K 值影響
    print("\n===== 開始分群分析 =====")
    # 執行不同數量的 KMeans 分群，觀察結果差異
    for n_clusters in [2, 3, 4, 5]:
        perform_clustering(customer_df.copy(), # 複製一份資料，避免不同 K 值的標籤互相影響
                           clustering_features,
                           n_clusters=n_clusters)
        # 在實際應用中，會使用 Elbow Method 或 Silhouette Score 來選擇最佳 K 值
        print("-" * 30) # 分隔不同 K 值的結果

    print("\n===== 分群分析結束 =====")
    print("\n\n") # 增加間隔

    # 執行分類任務
    print("\n===== 開始分類任務 =====")
    perform_classification(customer_df.copy(), # 複製一份資料
                           classification_regression_features,
                           classification_target)
    print("\n===== 分類任務結束 =====")
    print("\n\n") # 增加間隔

    # 執行回歸任務
    print("\n===== 開始回歸任務 =====")
    perform_regression(customer_df.copy(), # 複製一份資料
                       classification_regression_features,
                       regression_target)
    print("\n===== 回歸任務結束 =====")
    print("\n\n") # 增加間隔

    print("--- 程式執行完畢 ---")