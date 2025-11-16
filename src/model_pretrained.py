import tensorflow as tf
from tensorflow.keras import layers, regularizers


def build_model(input_shape=(224, 224, 3), weight_decay=1e-5, dropout=0.4):
    """
    EfficientNetB0 tabanlı binary classification modeli.
    
    NOT: 
    - Input görüntüler [0, 255] aralığında olmalı (uint8 veya float32)
    - preprocess_input katmanı görüntüleri EfficientNet'in beklediği formata dönüştürür
    - Base model başlangıçta frozen (trainable=False)
    """
    base = tf.keras.applications.EfficientNetB0(
        include_top=False,
        input_shape=input_shape,
        pooling='avg',
        weights='imagenet'
    )
    base.trainable = False  # freeze initially - Stage 1'de sadece dense katmanlar eğitilir

    inputs = layers.Input(shape=input_shape)
    # EfficientNet preprocessing: [0, 255] -> ImageNet normalization
    # Bu katman görüntüleri [-1, 1] aralığına normalize eder
    x = tf.keras.applications.efficientnet.preprocess_input(inputs)
    x = base(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(
        128,
        activation='relu',
        kernel_regularizer=regularizers.l2(weight_decay)
    )(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs, outputs, name="efficientnetb0_binary")
    return model
