"""
Simplified GradCAM for binary (sigmoid) models with Conv2D layers only.
"""

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime


class CustomGradCAM:
    def __init__(self, model, layer_name=None, log_file=None, debug_every=1):
        # Store original model for probability prediction
        self.original_model = model
        
        # Clone model and make last activation linear for GradCAM
        self.model = tf.keras.models.clone_model(model)
        self.model.set_weights(model.get_weights())
        last_layer = self.model.layers[-1]
        if hasattr(last_layer, "activation"):
            last_layer.activation = tf.keras.activations.linear

        self.layer_name = layer_name
        self.log_file = log_file
        self.debug_every = debug_every
        self._counter = 0

        # 🔍 1️⃣ Eğer layer adı yoksa ya da üst modelde bulunamadıysa:
        base_model = None
        base_model_input_layer = None
        
        # Find base model and its input layer (which comes after preprocessing)
        for i, l in enumerate(model.layers):
            if isinstance(l, tf.keras.Model):
                base_model = l
                # Base model'in input'u, preprocessing'in output'u olmalı
                # Model yapısını takip ederek base model'in input'unu bul
                if i > 0:
                    # Base model'den önceki layer preprocessing olmalı
                    prev_layer = model.layers[i-1]
                    base_model_input_layer = prev_layer.output
                break
        
        # If layer name not in top-level, check submodel
        if self.layer_name is None or self.layer_name not in [l.name for l in model.layers]:
            if base_model is not None:
                # Check if layer is in base model
                if self.layer_name and self.layer_name in [x.name for x in base_model.layers]:
                    print(f"[GradCAM] Found layer '{self.layer_name}' in submodel '{base_model.name}'")
                    # Build grad_model by following the model structure
                    # Use functional API to properly connect preprocessing -> base -> target layer
                    x = model.input
                    
                    # Find and apply preprocessing (layer before base model)
                    for i, l in enumerate(model.layers):
                        if isinstance(l, tf.keras.Model) and l == base_model:
                            # Apply all layers before base model (including preprocessing)
                            for j in range(i):
                                x = model.layers[j](x)
                            break
                    
                    # Now x is the preprocessed input (shape matches base_model.input)
                    # We need to get the target layer output from base model
                    # The solution: rebuild the path using functional API
                    # We'll create a model that takes model.input, applies preprocessing,
                    # then traces through base_model to the target layer
                    base_target_layer = base_model.get_layer(self.layer_name)
                    
                    # Rebuild the path: model.input -> preprocessing -> base_model -> target_layer
                    # We already have x = preprocessing(model.input)
                    # The key insight: We need to create a model that:
                    # 1. Takes model.input
                    # 2. Applies preprocessing (x)
                    # 3. Traces through base_model to target_layer
                    # 
                    # Since base_model is a functional model, we can create an intermediate model
                    # that maps base_model.input -> target_layer.output, but we need to connect
                    # it properly to x (which is preprocessing output)
                    #
                    # The solution: Create the intermediate model first, then use it in a
                    # functional way that properly connects x to it
                    intermediate_base = tf.keras.Model(
                        inputs=base_model.input,
                        outputs=base_target_layer.output
                    )
                    
                    # Now we need to call intermediate_base with x
                    # Since x has the same shape as base_model.input, we can do this
                    # But we need to ensure proper graph connection
                    # We'll use a Lambda layer that calls the model
                    # Note: The Lambda will properly handle the graph connection
                    def call_intermediate(inp):
                        return intermediate_base(inp)
                    
                    conv_output = tf.keras.layers.Lambda(
                        call_intermediate,
                        name='gradcam_intermediate'
                    )(x)
                    
                    # Get final model output
                    final_output = model.output
                    
                    self.grad_model = tf.keras.Model(
                        inputs=model.input,
                        outputs=[conv_output, final_output],
                    )
                elif base_model is not None:
                    # Auto-select last conv layer
                    conv_layers = [x.name for x in base_model.layers if "conv" in x.name.lower()]
                    if conv_layers:
                        self.layer_name = conv_layers[-1]
                        print(f"[GradCAM] Using last conv layer automatically: {self.layer_name}")
                        
                        # Build grad_model - same approach as above
                        x = model.input
                        for i, l in enumerate(model.layers):
                            if isinstance(l, tf.keras.Model) and l == base_model:
                                for j in range(i):
                                    x = model.layers[j](x)
                                break
                        
                        base_target_layer = base_model.get_layer(self.layer_name)
                        
                        # Create intermediate model from base_model that outputs target layer
                        intermediate_base = tf.keras.Model(
                            inputs=base_model.input,
                            outputs=base_target_layer.output
                        )
                        
                        # Use Lambda to call intermediate_base with x
                        def call_intermediate_auto(inp):
                            return intermediate_base(inp)
                        
                        conv_output = tf.keras.layers.Lambda(
                            call_intermediate_auto,
                            name='gradcam_intermediate_auto'
                        )(x)
                        final_output = model.output
                        
                        self.grad_model = tf.keras.Model(
                            inputs=model.input,
                            outputs=[conv_output, final_output],
                        )
                    else:
                        raise ValueError(
                            f"[GradCAM] No convolutional layer found inside submodel {base_model.name}"
                        )
            else:
                # No submodel found, try to find conv layer in top-level model
                conv_layers = [x.name for x in model.layers if isinstance(x, tf.keras.layers.Conv2D)]
                if conv_layers:
                    self.layer_name = conv_layers[-1]
                    print(f"[GradCAM] Using last conv layer in top model: {self.layer_name}")
                    self.grad_model = tf.keras.Model(
                        inputs=model.input,
                        outputs=[
                            model.get_layer(self.layer_name).output,
                            model.output,
                        ],
                    )
                else:
                    raise ValueError("[GradCAM] No convolutional layer found in model")
        else:
            # Layer üst modeldeyse normal davran
            self.grad_model = tf.keras.Model(
                inputs=model.input,
                outputs=[
                    model.get_layer(self.layer_name).output,
                    model.output,
                ],
            )

        print(f"[GradCAM] Using layer: {self.layer_name}")

    def _log(self, msg):
        print(msg)
        if self.log_file:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(f"{datetime.now():%Y-%m-%d %H:%M:%S} - {msg}\n")

    def compute_heatmap(self, image, class_idx=0):
        """Compute GradCAM heatmap for a single image."""
        if image.ndim == 3:
            image = np.expand_dims(image, 0)
        
        # Ensure image is in correct format [0, 255] for EfficientNet preprocessing
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255.0).astype(np.uint8)
            else:
                image = np.clip(image, 0, 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)

        # Convert to tensor
        image_tensor = tf.constant(image, dtype=tf.float32)

        with tf.GradientTape() as tape:
            conv_out, logits = self.grad_model(image_tensor, training=False)
            # class 1 → +logit, class 0 → -logit
            class_channel = logits[:, 0] if class_idx == 1 else -logits[:, 0]

        grads = tape.gradient(class_channel, conv_out)
        if grads is None:
            self._log("[GradCAM] Gradient is None, returning zeros.")
            h, w = int(conv_out.shape[1]), int(conv_out.shape[2])
            return np.zeros((h, w), dtype=np.float32)

        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_out = conv_out[0]  # (H, W, C)
        # Weighted sum across channels
        heatmap = tf.tensordot(conv_out, pooled_grads, axes=[[2], [0]])
        #heatmap = tf.reduce_sum(conv_out * pooled_grads[tf.newaxis, tf.newaxis, :], axis=-1)
        heatmap = tf.nn.relu(heatmap)
        heatmap = heatmap - tf.reduce_min(heatmap)
        heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)

        if self._counter % self.debug_every == 0:
            self._log(f"[DEBUG #{self._counter}] grad μ={tf.reduce_mean(grads):.6f}, "
                      f"heatmap σ={tf.math.reduce_std(heatmap):.6f}")
        self._counter += 1

        return heatmap.numpy()

    def overlay_heatmap(self, heatmap, image, alpha=0.4):
        """Overlay heatmap on image."""
        H, W = image.shape[:2]
        heatmap = tf.image.resize(heatmap[..., None], (H, W))[..., 0].numpy()
        heatmap = np.clip(heatmap, 0, 1)

        img = image.astype(np.float32)
        if img.max() > 1.0:
            img = img / 255.0

        cmap = plt.cm.get_cmap("jet")
        heatmap_rgb = cmap(heatmap)[..., :3]
        overlay = (1 - alpha) * img + alpha * heatmap_rgb
        return np.clip(overlay, 0, 1)

    def visualize(self, image, class_names=('NotDrowsy', 'Drowsy'),
                  true_idx=None, save_path=None):
        """Generate GradCAM visualization."""
        # Store original for display
        image_orig = image.copy()
        
        # Ensure image is in [0, 255] range for model prediction
        if image.max() <= 1.0:
            image_display = (image * 255.0).astype(np.uint8)
        else:
            image_display = np.clip(image, 0, 255).astype(np.uint8)
        
        if image.ndim == 3:
            image_batch = np.expand_dims(image_display, 0)
        else:
            image_batch = image_display

        # Use original model for probability prediction (with sigmoid activation)
        # Model expects [0, 255] range and handles preprocessing internally
        preds = self.original_model.predict(image_batch.astype(np.float32), verbose=0)
        prob = float(preds[0][0])
        pred_cls = 1 if prob >= 0.5 else 0
        disp_prob = prob if pred_cls == 1 else (1.0 - prob)

        heatmap = self.compute_heatmap(image_display, class_idx=pred_cls)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        # Display original image (convert back to [0, 1] for display if needed)
        display_img = image_orig.copy()
        if display_img.max() > 1.0:
            display_img = display_img / 255.0
        axes[0].imshow(display_img)
        txt = f"Pred: {class_names[pred_cls]} ({disp_prob:.3f})"
        if true_idx is not None:
            txt = f"Truth: {class_names[int(true_idx)]} | " + txt
        axes[0].text(5, 15, txt, color='white', fontsize=10,
                     bbox=dict(facecolor='black', alpha=0.6))
        axes[0].set_title("Original")
        axes[0].axis("off")

        im1 = axes[1].imshow(heatmap, cmap="jet")
        axes[1].set_title("Heatmap")
        axes[1].axis("off")
        plt.colorbar(im1, ax=axes[1])

        overlay = self.overlay_heatmap(heatmap, image_orig)
        axes[2].imshow(overlay)
        axes[2].set_title("Overlay")
        axes[2].axis("off")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            self._log(f"[GradCAM] Saved: {save_path}")
        plt.close(fig)
        return heatmap
