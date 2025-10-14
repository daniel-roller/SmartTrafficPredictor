# train_prophet_light.py（最終長期預測強化版）
import os
import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASET_DIR = os.path.join(BASE_DIR, "datasets")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

from config import (
    AGG_FREQ_LONG,
    LONG_TARGET,
    LONG_HORIZON_DAYS,
    BACKTEST_HORIZON_DAYS,
    BACKTEST_STEPS,
    USE_REGRESSORS,
    TREND_MODE,
    CAP_MULTIPLIER,
    FLOOR_MULTIPLIER,
    SMOOTHNESS
)

# ===============================
# 載入每日資料
# ===============================
def load_daily_series():
    csv_path = os.path.join(DATASET_DIR, "daily_series.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"❌ 找不到 {csv_path}，請先執行 make_dataset.py 產生日資料")

    df = pd.read_csv(csv_path, parse_dates=["ds"])
    df["ds"] = pd.to_datetime(df["ds"]).dt.tz_localize(None)

    if "y" not in df.columns:
        raise ValueError("❌ 缺少欄位 y，請確認 daily_series.csv 正確生成")

    df.fillna(0, inplace=True)
    df["cap"] = df["y"].max() * CAP_MULTIPLIER
    df["floor"] = df["y"].min() * FLOOR_MULTIPLIER

    return df

# ===============================
# 建立 Prophet 模型
# ===============================
def fit_prophet(df):
    m = Prophet(
        growth=TREND_MODE,
        daily_seasonality="auto",
        weekly_seasonality="auto",
        yearly_seasonality="auto",
        seasonality_mode="additive",
        changepoint_prior_scale=SMOOTHNESS
    )

    # 加入台灣假期
    try:
        m.add_country_holidays(country_name="TW")
    except Exception:
        print("⚠️ 無法載入台灣假期，但不影響模型運作")

    # === NEW: 加入月與季度週期 ===
    m.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    m.add_seasonality(name='quarterly', period=91.25, fourier_order=5)

    # === NEW: 額外特徵（regressors） ===
    regressors = [
        "total_vehicles", "is_weekend", "is_peak",
        "hour_sin", "hour_cos", "weekday_sin", "weekday_cos", "speed_ma7"
    ]
    if USE_REGRESSORS:
        for reg in regressors:
            if reg in df.columns:
                m.add_regressor(reg)
    
    m.fit(df)
    return m

# ===============================
# 預測未來 N 天
# ===============================
def forecast_future(m, df, periods):
    future = m.make_future_dataframe(periods=periods, freq="D")
    future["cap"] = df["cap"].iloc[0]
    future["floor"] = df["floor"].iloc[0]

    regressors = [
        "total_vehicles", "is_weekend", "is_peak",
        "hour_sin", "hour_cos", "weekday_sin", "weekday_cos", "speed_ma7"
    ]
    for reg in regressors:
        if reg in df.columns:
            future[reg] = df[reg].iloc[-1]
    
    forecast = m.predict(future)
    return forecast

# ===============================
# 畫圖
# ===============================
def plot_forecast(m, fcst, out_png):
    fig = m.plot(fcst)
    plt.title(f"Prophet Forecast ({LONG_HORIZON_DAYS} days, trend={TREND_MODE})")
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

# ===============================
# 滾動回測
# ===============================
def rolling_backtest(df, horizon_days=30, steps=3):
    results = []
    last_day = df["ds"].max()
    for i in range(steps, 0, -1):
        cutoff = last_day - pd.Timedelta(days=horizon_days * i)
        train_df = df[df["ds"] <= cutoff].copy()
        test_df = df[(df["ds"] > cutoff) & (df["ds"] <= cutoff + pd.Timedelta(days=horizon_days))].copy()

        if len(train_df) < 90 or len(test_df) < 10:
            continue

        m = fit_prophet(train_df)
        future = m.make_future_dataframe(periods=horizon_days, freq="D")
        future["cap"] = train_df["cap"].iloc[0]
        future["floor"] = train_df["floor"].iloc[0]

        regressors = [
            "total_vehicles", "is_weekend", "is_peak",
            "hour_sin", "hour_cos", "weekday_sin", "weekday_cos", "speed_ma7"
        ]
        for reg in regressors:
            if reg in train_df.columns:
                future[reg] = train_df[reg].iloc[-1]

        fcst = m.predict(future)
        merged = pd.merge(test_df[["ds", "y"]], fcst[["ds", "yhat"]], on="ds", how="inner")
        if merged.empty:
            continue

        mae = np.mean(np.abs(merged["y"] - merged["yhat"]))
        rmse = np.sqrt(np.mean((merged["y"] - merged["yhat"])**2))
        results.append({"fold_cutoff": cutoff.date(), "MAE": mae, "RMSE": rmse})

    return results

# ===============================
# 主程式
# ===============================
def main():
    df = load_daily_series()

    horizons = [30, 60, 90]  # ⬅️ 新增：要比較的預測天數
    out_dir = os.path.join(RESULTS_DIR, "prophet_compare_horizons")
    os.makedirs(out_dir, exist_ok=True)

    all_forecasts = []

    for horizon in horizons:
        print(f"\n🚀 正在預測 {horizon} 天...\n")

        m = fit_prophet(df)
        fcst = forecast_future(m, df, horizon)

        # 儲存結果
        all_forecasts.append((horizon, fcst))
        metrics_path = os.path.join(out_dir, f"metrics_prophet_{horizon}d.txt")

        # 簡單誤差評估
        y_true = df["y"].values[-min(len(df), horizon):].astype(float)
        y_pred = fcst["yhat"].values[-min(len(df), horizon):].astype(float)
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

        with open(metrics_path, "w") as f:
            f.write(f"Prophet {horizon}d Forecast\nMAE={mae:.3f}, RMSE={rmse:.3f}\n")

        print(f"✅ 預測 {horizon} 天完成：MAE={mae:.3f}, RMSE={rmse:.3f}")

    # --- 畫比較圖 ---
    plt.figure(figsize=(10, 6))
    plt.plot(df["ds"], df["y"], label="True", color="black")

    colors = ["tab:blue", "tab:orange", "tab:green"]
    for (horizon, fcst), c in zip(all_forecasts, colors):
        plt.plot(fcst["ds"], fcst["yhat"], label=f"{horizon}-day Forecast", color=c)

    plt.title("Prophet Forecast Comparison (30 / 60 / 90 days)")
    plt.xlabel("Date")
    plt.ylabel("Average Speed (km/h)")
    plt.legend()
    out_png = os.path.join(out_dir, "prophet_compare_forecasts.png")
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()
    print(f"\n📊 已輸出比較圖：{out_png}\n")

    out_dir = os.path.join(RESULTS_DIR, f"prophet_{LONG_HORIZON_DAYS}d_fixed")
    os.makedirs(out_dir, exist_ok=True)

    df = load_daily_series()

    m_all = fit_prophet(df)
    fcst_all = forecast_future(m_all, df, LONG_HORIZON_DAYS)

    out_png = os.path.join(out_dir, f"prophet_forecast_{LONG_HORIZON_DAYS}d_fixed.png")
    plot_forecast(m_all, fcst_all, out_png)

    bt = rolling_backtest(df, horizon_days=BACKTEST_HORIZON_DAYS, steps=BACKTEST_STEPS)

    metrics_path = os.path.join(out_dir, f"metrics_prophet_{LONG_HORIZON_DAYS}d_fixed.txt")
    with open(metrics_path, "w", encoding="utf-8") as f:
        if bt:
            avg_mae = np.mean([r["MAE"] for r in bt])
            avg_rmse = np.mean([r["RMSE"] for r in bt])
            f.write(f"Backtest ({BACKTEST_STEPS} folds × {BACKTEST_HORIZON_DAYS} days)\n")
            for r in bt:
                f.write(f"- cutoff={r['fold_cutoff']}, MAE={r['MAE']:.3f}, RMSE={r['RMSE']:.3f}\n")
            f.write(f"\nBacktest Avg: MAE={avg_mae:.3f}, RMSE={avg_rmse:.3f}\n")
        else:
            f.write("Backtest not available (data too short)\n")

    print("✅ Prophet 長期預測完成")
    print("  圖片：", out_png)
    print("  指標：", metrics_path)

if __name__ == "__main__":
    main()
