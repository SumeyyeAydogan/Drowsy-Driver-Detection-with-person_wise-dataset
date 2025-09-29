import os, re, shutil, random
from collections import defaultdict
from typing import Optional, Dict, List, Set, Tuple

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
SUBJECT_RE = re.compile(r"^([A-Za-z]+)")

def _subject_from_name(filename: str) -> Optional[str]:
    name, _ = os.path.splitext(filename)
    m = SUBJECT_RE.match(name)
    if not m:
        return None
    return m.group(1).lower()

def _list_images(cls_dir: str):
    for f in os.listdir(cls_dir):
        p = os.path.join(cls_dir, f)
        if os.path.isfile(p) and os.path.splitext(f)[1].lower() in EXTS:
            yield f

def _split_subjects(subjects: List[str], ratios: Dict[str, float], seed: int) -> Dict[str, Set[str]]:
    rng = random.Random(seed)
    subjects = subjects[:]
    rng.shuffle(subjects)
    n = len(subjects)
    n_train = int(ratios["train"] * n)
    n_val   = int(ratios["val"] * n)
    train = subjects[:n_train]
    val   = subjects[n_train:n_train+n_val]
    test  = subjects[n_train+n_val:]
    return {"train": set(train), "val": set(val), "test": set(test)}

def split_new_dataset(
    raw_directory: str,
    output_directory: str,
    classes: Tuple[str, str] = ("Drowsy", "NotDrowsy"),
    split_ratios: Optional[Dict[str, float]] = None,
    seed: int = 42,
    verbose: bool = True,
):
    """
    Person-wise train/val/test split per class and copy files into
    output_directory/{train,val,test}/{class}.

    Returns a summary dict with image counts per split/class.
    """
    ratios = split_ratios or {"train": 0.7, "val": 0.15, "test": 0.15}

    # Reset destination dir if exists (simple clean)
    if os.path.isdir(output_directory):
        if verbose:
            print(f"[INFO] Removing existing output directory: {output_directory}")
        shutil.rmtree(output_directory)

    # Prepare destination dirs
    for split in ("train", "val", "test"):
        for cls in classes:
            os.makedirs(os.path.join(output_directory, split, cls), exist_ok=True)

    # Build subject->files per class
    per_class_subject_files: Dict[str, Dict[str, List[str]]] = {}
    for cls in classes:
        cls_src = os.path.join(raw_directory, cls)
        subj2files = defaultdict(list)
        if not os.path.isdir(cls_src):
            raise FileNotFoundError(f"Class directory not found: {cls_src}")
        for fname in _list_images(cls_src):
            subj = _subject_from_name(fname)
            if subj is None:
                if verbose:
                    print(f"[WARN] Subject not parsed, skipping: {fname}")
                continue
            subj2files[subj].append(fname)
        per_class_subject_files[cls] = subj2files

    # Split subjects per class
    per_class_splits: Dict[str, Dict[str, Set[str]]] = {}
    for cls, subj2files in per_class_subject_files.items():
        subjects = list(subj2files.keys())
        per_class_splits[cls] = _split_subjects(subjects, ratios, seed)

    # Copy files
    for cls, subj2files in per_class_subject_files.items():
        src_dir = os.path.join(raw_directory, cls)
        for split, subj_set in per_class_splits[cls].items():
            dst_dir = os.path.join(output_directory, split, cls)
            for subj in subj_set:
                for fname in subj2files[subj]:
                    shutil.copy(os.path.join(src_dir, fname), os.path.join(dst_dir, fname))

    # Summary
    summary: Dict[str, Dict[str, int]] = {"train": {}, "val": {}, "test": {}}
    for split in ("train", "val", "test"):
        for cls in classes:
            d = os.path.join(output_directory, split, cls)
            count = len([f for f in os.listdir(d) if os.path.isfile(os.path.join(d, f))])
            summary[split][cls] = count
    if verbose:
        print("=== Split summary ===")
        for split in ("train", "val", "test"):
            print(f"{split:5s} -> " + ", ".join(f"{cls}:{summary[split][cls]}" for cls in classes))
    return summary
