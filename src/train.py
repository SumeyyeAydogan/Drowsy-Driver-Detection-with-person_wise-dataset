import tensorflow as tf
from tensorflow.keras.metrics import BinaryAccuracy, Precision, Recall, AUC
import copy


def train_model(model, train_ds, val_ds, epochs=15, callbacks_stage1=None, callbacks_stage2=None):
    """
    İki aşamalı eğitim:
      1️⃣ Base (EfficientNet) freeze edilir, yalnızca dense katmanlar eğitilir.
      2️⃣ Fine-tuning aşamasında base'in son 30 katmanı açılır.
    """

    # --- 1️⃣ Freeze stage ---
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=[
            BinaryAccuracy(name="accuracy"),
            Precision(name="precision"),
            Recall(name="recall"),
            AUC(name="auc"),
        ],
    )

    print("🔒 Stage 1: Training with frozen base...")
    history_1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=5,
        callbacks=callbacks_stage1,
        verbose=1,
    )

    # --- 2️⃣ Fine-tuning stage ---
    print("🔓 Stage 2: Fine-tuning last 30 layers of EfficientNet base...")
    # Find the EfficientNet base model (it's a submodel)
    base_model = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and 'efficientnet' in layer.name.lower():
            base_model = layer
            break
    
    if base_model is None:
        # Try to find by name
        try:
            base_model = model.get_layer("efficientnetb0")
        except:
            # Last resort: find first Model layer
            for layer in model.layers:
                if isinstance(layer, tf.keras.Model):
                    base_model = layer
                    break
    
    if base_model is None:
        raise ValueError("Could not find EfficientNet base model in the model structure")
    
    print(f"🔓 Found base model: {base_model.name}")
    
    # Unfreeze last 30 layers (including BatchNormalization for fine-tuning)
    # BatchNorm'ları da unfreeze etmek fine-tuning için önemli
    layers_to_unfreeze = base_model.layers[-30:]
    for layer in layers_to_unfreeze:
        layer.trainable = True
    
    num_open = sum(l.trainable for l in base_model.layers)
    num_total = len(base_model.layers)
    print(f"🔓 Fine-tuning {num_open}/{num_total} EfficientNet layers")
    print(f"   (Unfrozen layers: {[l.name for l in layers_to_unfreeze[:5]]} ... {[l.name for l in layers_to_unfreeze[-2:]]})")

    # Re-compile with lower learning rate for fine-tuning (best practice)
    # Fine-tuning için genelde 10x daha düşük LR kullanılır
    fine_tune_lr = 1e-5  # Stage 1'de 1e-4, Stage 2'de 1e-5
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=fine_tune_lr, clipnorm=1.0),
        loss="binary_crossentropy",
        metrics=[
            BinaryAccuracy(name="accuracy"),
            Precision(name="precision"),
            Recall(name="recall"),
            AUC(name="auc"),
        ],
    )

    history_2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        initial_epoch=5,
        callbacks=callbacks_stage2,
        verbose=1,
    )

    # --- 3️⃣ History'leri birleştir ---
    full_history = {}
    for k in set(history_1.history.keys()).union(history_2.history.keys()):
        h1 = [float(v.numpy() if hasattr(v, "numpy") else v)
            for v in history_1.history.get(k, [])]
        h2 = [float(v.numpy() if hasattr(v, "numpy") else v)
            for v in history_2.history.get(k, [])]
        full_history[k] = h1 + h2
    return type("MergedHistory", (), {"history": full_history})()
