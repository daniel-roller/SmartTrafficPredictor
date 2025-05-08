# data_preprocessing.py

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression

RANDOM_STATE = 42
N_SAMPLES = 1000


def simulate_customer_data(n_samples=N_SAMPLES, random_state=RANDOM_STATE):
    """
    模擬顧客資料，包括分類目標 Segment、回歸目標 PurchaseAmount，以及其他特徵。
    """
    # 分類資料：預測 Segment 用
    X_clf, y_clf = make_classification(
        n_samples=n_samples,
        n_features=4,
        n_informative=3,
        n_redundant=1,
        n_classes=3,
        n_clusters_per_class=1,
        flip_y=0.01,
        random_state=random_state
    )

    # 回歸資料：預測購買金額用
    X_reg, y_reg = make_regression(
        n_samples=n_samples,
        n_features=4,
        n_informative=4,
        noise=10.0,
        random_state=random_state
    )

    # 組合成 DataFrame
    df = pd.DataFrame(X_clf, columns=['Age', 'Income', 'PurchaseFreq', 'Membership'])

    df['Age'] = (df['Age'] * 5 + 30).astype(int)
    df['Income'] = (df['Income'] * 5000 + 50000).astype(int)
    df['PurchaseFreq'] = (df['PurchaseFreq'] * 2 + 5).astype(int).clip(1, 20)
    df['Membership'] = (df['Membership'] > 0).astype(int)

    segment_map = {0: 'Segment A', 1: 'Segment B', 2: 'Segment C'}
    df['Segment'] = pd.Series(y_clf).map(segment_map)

    df['PurchaseAmount'] = (y_reg + abs(y_reg.min()) + 50).astype(int).clip(lower=50)
    df['Gender'] = np.random.choice(['Male', 'Female', 'Other'], size=n_samples)

    return df
