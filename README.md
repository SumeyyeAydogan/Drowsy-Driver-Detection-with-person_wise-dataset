# DDD-Project (Person-wise Drowsiness Detection)

A PyTorch-based project for person-wise drowsiness detection from images. The repository includes training, evaluation, Grad-CAM visualization, dataset splitting utilities, and export tools.

## Features
- Person-wise train/val/test splitting utilities
- Configurable training loop with callbacks and checkpoints
- Evaluation scripts with metrics and plots
- Grad-CAM visualization for model interpretability
- Export to common model formats (e.g., ONNX)

## Repository Structure
- `src/` – core modules: dataloaders, model, training loop, callbacks, evaluation, export, utils
- `scripts/` – entry-point scripts (train, eval, grad-cam, dataset split, etc.)
- `dataset/` – raw dataset folder (ignored by Git)
- `old_splitted_dataset/` – historical splits (ignored by Git)
- `runs/` – experiment outputs, logs, and checkpoints (ignored by Git)
- `main.py` – optional top-level entry

Note: Large data and run artifacts are excluded via `.gitignore`.

## Requirements
- Python 3.9+
- PyTorch and torchvision compatible with your CUDA/CPU setup
- Common scientific stack: numpy, pandas, matplotlib, scikit-learn, tqdm

Install dependencies (example):
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install --upgrade pip
pip install torch torchvision torchaudio  # choose versions per your CUDA
pip install -r requirements.txt  # if provided
```
If `requirements.txt` is not present, install packages listed in the source imports as needed.

## Data Preparation
Place your data under `dataset/` using a structure such as:
dataset/
Drowsy/
<images>.png
NotDrowsy/
<images>.png

Utilities for splitting the dataset person-wise are provided under `src/` (and/or `scripts/`). Adjust paths in scripts as needed.

## Training
Example command (adjust arguments to your needs):
```bash
python scripts/train.py \
  --data_dir dataset \
  --epochs 20 \
  --batch_size 32 \
  --lr 3e-4 \
  --output_dir runs/exp1
```
Check `src/train.py` and `src/run_manager.py` for available flags and defaults.

## Evaluation
Run evaluation on a trained checkpoint:
```bash
python scripts/eval_script.py \
  --data_dir dataset \
  --checkpoint runs/exp1/best.ckpt \
  --out_dir runs/exp1/eval
```
This will save metrics and plots under the specified output directory.

## Grad-CAM Visualization
Generate Grad-CAM heatmaps to interpret predictions:
```bash
python scripts/gradcam_simple.py \
  --checkpoint runs/exp1/best.ckpt \
  --image_path path/to/sample.png \
  --out_dir runs/exp1/gradcam
```
See `src/gradcam.py` for more options.

## Exporting Models
Export a trained model (e.g., ONNX):
```bash
python scripts/export.py \
  --checkpoint runs/exp1/best.ckpt \
  --export_path runs/exp1/model.onnx
```

## Reproducibility
- Set random seeds where possible (see `src/utils.py`).
- Keep track of experiment configs and versions in `runs/`.

## Git and Large Files
- `.gitignore` excludes `dataset/`, `runs/`, checkpoints, and large artifacts by default.
- If you need to version large model files, consider Git LFS:
```bash
git lfs install
git lfs track "*.pt" "*.pth" "*.ckpt" "*.onnx"
```

## License
Specify a license if applicable (e.g., MIT). Add a `LICENSE` file to the root of the repository.

## Citation
If you use this repository in academic work, please cite it appropriately. Add BibTeX or references here as needed.

## Contact
For questions or contributions, please open an issue or submit a pull request.