# train_lstm.py
import os
import pickle
import glob
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

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
DATASET_PATH = pkl_files[-1]  # 選最新的一個
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

# === 建立 LSTM 模型 ===
model = models.Sequential([
    layers.Input(shape=(time_steps, n_features)),
    layers.LSTM(64, return_sequences=False),
    layers.Dropout(0.2),
    layers.Dense(n_outputs)
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss=tf.keras.losses.MeanSquaredError(),
    metrics=[tf.keras.metrics.MeanAbsoluteError()]
)

model.summary()

# === 訓練 ===
ckpt_path = os.path.join(MODEL_DIR, "lstm_baseline.best.h5")
callbacks_list = [
    callbacks.EarlyStopping(patience=10, restore_best_weights=True),
    callbacks.ModelCheckpoint(ckpt_path, save_best_only=True)
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
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
plt.savefig(os.path.join(RESULTS_DIR, "train_loss.png"))
plt.close()

# === 儲存驗證集結果 ===
val_loss, val_mae = model.evaluate(X_val, y_val, verbose=0)
with open(os.path.join(RESULTS_DIR, "metrics_val.txt"), "w") as f:
    f.write(f"Validation Loss (MSE): {val_loss:.4f}\n")
    f.write(f"Validation MAE: {val_mae:.4f}\n")

print("✅ 訓練完成，最佳模型存到:", ckpt_path)
