import os
import tensorflow as tf
from src.model import build_model
from src.run_manager import RunManager
from src.dataloader import get_binary_pipelines
from src.evaluate import evaluate_model

# 1) Run directory (contains checkpoints)
# Example: runs/old/30_e-5_new_mid-aug
base_run_dir = "runs/50_epoch_e-5--"  # set your run folder here

# 2) Build model with the same architecture
model = build_model()

# 3) Load latest checkpoint via RunManager
# RunManager expects run_name relative to the "runs" folder
run_name = os.path.relpath(base_run_dir, start="runs")  # e.g., "old/30_e-5_new_mid-aug"
rm = RunManager(run_name=run_name)
start_epoch = rm.load_latest_checkpoint(model)  # returns 0 if not found

# 4) Save final model (models/final_model.h5)
rm.save_final_model(model)

# 5) Datasets
split_root = "splitted_dataset"  # your split root
train_ds, val_ds, test_ds, class_names = get_binary_pipelines(
    split_root,
    img_size=(224, 224),
    batch_size=32,
    seed=42
)

# 6) Evaluate (plots saved under base_run_dir)
plots_dir = os.path.join(base_run_dir, "plots", "eval")
os.makedirs(plots_dir, exist_ok=True)

metrics = evaluate_model(
    model,
    test_ds,
    plots_dir=plots_dir,
    class_names=class_names,
    subject_diverse_dir=None,
    misclassified_only=False,
    ds_name="test"
)
# (Optional) Evaluate on train/val as well
print("Evaluating model on train and val sets as well...")
train_plots_dir = os.path.join(base_run_dir, "plots", "train_eval")
val_plots_dir = os.path.join(base_run_dir, "plots", "val_eval")
os.makedirs(train_plots_dir, exist_ok=True)
os.makedirs(val_plots_dir, exist_ok=True)

evaluate_model(
    model,
    train_ds,
    plots_dir=train_plots_dir,
    class_names=class_names,
    subject_diverse_dir=os.path.join(split_root, "train"),
    misclassified_only=True,
    ds_name="train"
)

evaluate_model(
    model,
    val_ds,
    plots_dir=val_plots_dir,
    class_names=class_names,
    subject_diverse_dir=os.path.join(split_root, "val"),
    misclassified_only=False,
    ds_name="val"
)

print("Done. Final model and evaluation plots saved.")