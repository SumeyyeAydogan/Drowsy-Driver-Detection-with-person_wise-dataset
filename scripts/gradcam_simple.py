"""
Simple GradCAM Usage Example
"""
import os
import numpy as np
import tensorflow as tf
from src.model import build_model
from src.gradcam import GradCAM, analyze_model_gradcam, _pred_to_prob_and_class
from src.dataloader import get_binary_pipelines


def main():
    """
    Simple GradCAM example
    """
    print("🎯 Simple GradCAM Example")
    print("=" * 40)
    
    # Configure output directories once
    base_run_dir = "runs/old/30_e-5_new_mid-aug"
    plots_dir = os.path.join(base_run_dir, "plots", "train", "gradcam")
    os.makedirs(plots_dir, exist_ok=True)

    # 1. Load or create model
    print("1. Loading model...")
    try:
        # Try to load a trained model
        model_path = os.path.join(base_run_dir, "models", "final_model.h5")
        model = tf.keras.models.load_model(model_path)
        print(f"✅ Loaded trained model: {model_path}")
    except:
        # Create new model if no trained model found
        model = build_model()
        print("⚠️  Using untrained model")
    
    # 2. Load test data
    print("2. Loading test data...")
    try:
        train_ds, val_ds, test_ds, class_names = get_binary_pipelines(
        "old_splitted_dataset",
        img_size=(224, 224),
        batch_size=32,
        seed=42
    )
        print("✅ Test data loaded")
    except:
        print("❌ Error loading test data")
        return
    
    # 3. Single image example
    print("3. Single image GradCAM...")
    for batch_images, batch_labels in train_ds.take(1):
        sample_image = batch_images[0].numpy()
        true_label = batch_labels[0].numpy()
        # Ensure true_label is an integer class index (handle one-hot or scalar)
        try:
            true_label_idx = int(np.argmax(true_label)) if np.ndim(true_label) > 0 else int(true_label)
        except Exception:
            true_label_idx = int(true_label)
        break
    
    gradcam = GradCAM(model)
    fig, prediction = gradcam.visualize(
        sample_image, 
        class_names=class_names,
        save_path=os.path.join(plots_dir, "gradcam_example.png"),
        true_class_idx=true_label_idx
    )
    
    prob, pred_class = _pred_to_prob_and_class(prediction)
    print(f"   True: {['Not Drowsy', 'Drowsy'][true_label_idx]}")
    print(f"   Pred: {['Not Drowsy', 'Drowsy'][pred_class]} ({prob:.3f})")
    
    # 4. Batch analysis
    print("4. Batch analysis...")
    analyze_model_gradcam(model, train_ds, num_samples=10, output_dir=plots_dir)
    
    print("\n✅ GradCAM example completed!")
    print(f"📁 Check 'gradcam_example.png' and {base_run_dir}/plots/train/gradcam/ directory")


if __name__ == "__main__":
    main()
