# train_lstm.py
import os
import pickle
import glob
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

# 匯入超參數設定
from config import (
    MODEL_TYPE, LSTM_UNITS, DROPOUT, LEARNING_RATE,
    BATCH_SIZE, EPOCHS, PATIENCE
)

layers = keras.layers
models = keras.models
callbacks = keras.callbacks

# === 自動偵測最新的 dataset ===
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASET_DIR = os.path.join(BASE_DIR, "datasets")
MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

pkl_files = sorted(glob.glob(os.path.join(DATASET_DIR, "traffic_lstm_*.pkl")))
if not pkl_files:
    raise FileNotFoundError("❌ 在 datasets/ 找不到任何 .pkl，請先執行 make_dataset.py")
DATASET_PATH = pkl_files[-1]
print(f"✅ 使用的 dataset: {DATASET_PATH}")

# === 載入資料 ===
with open(DATASET_PATH, "rb") as f:
    data = pickle.load(f)

X_train, y_train = data["X_train"], data["y_train"]
X_val, y_val     = data["X_val"], data["y_val"]
X_test, y_test   = data["X_test"], data["y_test"]

print("資料形狀：")
print("  X_train:", X_train.shape, "y_train:", y_train.shape)
print("  X_val:",   X_val.shape,   "y_val:",   y_val.shape)
print("  X_test:",  X_test.shape,  "y_test:",  y_test.shape)

time_steps  = X_train.shape[1]
n_features  = X_train.shape[2]
n_outputs   = y_train.shape[1]

# === 建立模型 ===
if MODEL_TYPE == "LSTM":
    model = models.Sequential([
        layers.Input(shape=(time_steps, n_features)),
        layers.LSTM(LSTM_UNITS),
        layers.Dropout(DROPOUT),
        layers.Dense(n_outputs)
    ])
elif MODEL_TYPE == "GRU":
    model = models.Sequential([
        layers.Input(shape=(time_steps, n_features)),
        layers.GRU(LSTM_UNITS),
        layers.Dropout(DROPOUT),
        layers.Dense(n_outputs)
    ])
elif MODEL_TYPE == "LSTM2":  # 雙層 LSTM
    model = models.Sequential([
        layers.Input(shape=(time_steps, n_features)),
        layers.LSTM(LSTM_UNITS, return_sequences=True),
        layers.LSTM(LSTM_UNITS),
        layers.Dropout(DROPOUT),
        layers.Dense(n_outputs)
    ])
elif MODEL_TYPE == "CNN_LSTM":
    model = models.Sequential([
        layers.Input(shape=(time_steps, n_features)),
        layers.Conv1D(64, kernel_size=3, activation="relu"),
        layers.MaxPooling1D(pool_size=2),
        layers.LSTM(LSTM_UNITS),
        layers.Dropout(DROPOUT),
        layers.Dense(n_outputs)
    ])
else:
    raise ValueError(f"❌ 不支援的 MODEL_TYPE: {MODEL_TYPE}")

# === 編譯模型 ===
model.compile(
    optimizer=keras.optimizers.Adam(LEARNING_RATE),
    loss=keras.losses.Huber(delta=1.0),     # ⬅️ 改這裡
    metrics=[keras.metrics.MeanAbsoluteError()]
)

model.summary()

# === 訓練 ===
ckpt_path = os.path.join(MODEL_DIR, f"{MODEL_TYPE.lower()}_baseline.best.h5")
callbacks_list = [
    callbacks.EarlyStopping(patience=PATIENCE, restore_best_weights=True),
    callbacks.ModelCheckpoint(ckpt_path, save_best_only=True)
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks_list,
    verbose=1
)

# === 儲存訓練曲線 ===
plt.figure()
plt.plot(history.history["loss"], label="train_loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE)")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(RESULTS_DIR, f"train_loss_{MODEL_TYPE}.png"))
plt.close()

# === 儲存驗證集結果 ===
val_loss, val_mae = model.evaluate(X_val, y_val, verbose=0)
with open(os.path.join(RESULTS_DIR, f"metrics_val_{MODEL_TYPE}.txt"), "w") as f:
    f.write(f"Validation Loss (MSE): {val_loss:.4f}\n")
    f.write(f"Validation MAE: {val_mae:.4f}\n")

print(f"✅ 訓練完成，最佳模型存到: {ckpt_path}")
