import sys
print(f"Python版本: {sys.version}")

try:
    import tensorflow as tf
    print(f"✅ TensorFlow版本: {tf.__version__}")
    
    from tensorflow.keras import models, layers, optimizers, callbacks
    print("✅ Keras模組匯入成功")
    
    # 測試建立簡單模型
    model = models.Sequential([
        layers.Dense(10, activation='relu', input_shape=(5,)),
        layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    print("✅ Keras模型建立成功")
    
except ImportError as e:
    print(f"❌ 匯入錯誤: {e}")
except Exception as e:
    print(f"❌ 其他錯誤: {e}")