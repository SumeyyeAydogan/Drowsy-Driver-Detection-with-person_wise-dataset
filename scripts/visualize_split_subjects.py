import os
import re
import argparse
from typing import Dict, Set, List, Tuple, DefaultDict
from collections import defaultdict
import matplotlib.pyplot as plt

SUBJECT_RE = re.compile(r"^([A-Za-z]+)")


def _subjects_from_name(filename: str):
    name, _ = os.path.splitext(filename)
    m = SUBJECT_RE.match(name)
    if not m:
        return None, None
    raw = m.group(1)
    return raw, raw.lower()


def collect_subjects(split_root: str, classes: Tuple[str, str] = ("NotDrowsy", "Drowsy")) -> Dict[str, Set[str]]:
    result: Dict[str, Set[str]] = {"train": set(), "val": set(), "test": set()}
    for split in ("train", "val", "test"):
        for cls in classes:
            cls_dir = os.path.join(split_root, split, cls)
            if not os.path.isdir(cls_dir):
                continue
            for fname in os.listdir(cls_dir):
                fpath = os.path.join(cls_dir, fname)
                if not os.path.isfile(fpath):
                    continue
                _, subj_norm = _subjects_from_name(fname)
                if subj_norm:
                    result[split].add(subj_norm)
    return result


def collect_subjects_with_cases(split_root: str, classes: Tuple[str, str] = ("NotDrowsy", "Drowsy")):
    # norm -> raw_case -> split -> count
    norm_to_case_split_counts: Dict[str, DefaultDict[str, DefaultDict[str, int]]] = {}
    for split in ("train", "val", "test"):
        for cls in classes:
            cls_dir = os.path.join(split_root, split, cls)
            if not os.path.isdir(cls_dir):
                continue
            for fname in os.listdir(cls_dir):
                fpath = os.path.join(cls_dir, fname)
                if not os.path.isfile(fpath):
                    continue
                subj_raw, subj_norm = _subjects_from_name(fname)
                if not subj_norm:
                    continue
                if subj_norm not in norm_to_case_split_counts:
                    norm_to_case_split_counts[subj_norm] = defaultdict(lambda: defaultdict(int))
                norm_to_case_split_counts[subj_norm][subj_raw][split] += 1
    return norm_to_case_split_counts


def print_overview(split_to_subjects: Dict[str, Set[str]]):
    print("=== Subjects per split ===")
    for split in ("train", "val", "test"):
        subs = sorted(split_to_subjects.get(split, set()))
        print(f"{split} ({len(subs)} subjects):")
        if subs:
            print("  " + ", ".join(subs))
        else:
            print("  (none)")


def export_csv(split_to_subjects: Dict[str, Set[str]], out_csv: str):
    # Build unified subject list
    all_subjects: Set[str] = set()
    for s in split_to_subjects.values():
        all_subjects |= s
    rows: List[str] = ["subject,split"]
    for subj in sorted(all_subjects):
        split = (
            "train" if subj in split_to_subjects["train"] else
            "val" if subj in split_to_subjects["val"] else
            "test" if subj in split_to_subjects["test"] else
            "unknown"
        )
        rows.append(f"{subj},{split}")
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("\n".join(rows))
    print(f"[SAVED] {out_csv}")


def export_pivot_csv(split_to_subjects: Dict[str, Set[str]], case_counts: Dict[str, DefaultDict[str, DefaultDict[str, int]]], out_csv: str):
    # All normalized subjects
    all_subjects: Set[str] = set()
    for s in split_to_subjects.values():
        all_subjects |= s
    rows: List[str] = ["subject,train_count,val_count,test_count"]
    for subj in sorted(all_subjects):
        # Sum counts from all case variants for each split
        train_c = val_c = test_c = 0
        if subj in case_counts:
            for _case, split_map in case_counts[subj].items():
                train_c += split_map.get("train", 0)
                val_c += split_map.get("val", 0)
                test_c += split_map.get("test", 0)
        rows.append(f"{subj},{train_c},{val_c},{test_c}")
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("\n".join(rows))
    print(f"[SAVED] {out_csv}")


def export_case_csv(case_counts: Dict[str, DefaultDict[str, DefaultDict[str, int]]], out_csv: str):
    rows: List[str] = ["subject_norm,case_variant,train_count,val_count,test_count,all_variants_same_split"]
    for subj_norm in sorted(case_counts.keys()):
        # Determine if all variants reside in the same split
        variant_splits: List[Set[str]] = []
        for case_var, split_map in case_counts[subj_norm].items():
            present_splits = {s for s, c in split_map.items() if c > 0}
            variant_splits.append(present_splits)
        union_splits: Set[str] = set().union(*variant_splits) if variant_splits else set()
        same_split = (len(union_splits) <= 1)
        for case_var, split_map in case_counts[subj_norm].items():
            tr = split_map.get("train", 0)
            va = split_map.get("val", 0)
            te = split_map.get("test", 0)
            rows.append(f"{subj_norm},{case_var},{tr},{va},{te},{'yes' if same_split else 'no'}")
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("\n".join(rows))
    print(f"[SAVED] {out_csv}")


def plot_counts(split_to_subjects: Dict[str, Set[str]], out_png: str = None):
    splits = ["train", "val", "test"]
    counts = [len(split_to_subjects.get(s, set())) for s in splits]

    plt.figure(figsize=(6, 4))
    bars = plt.bar(splits, counts, color=["#4CAF50", "#FFC107", "#03A9F4"])
    plt.title("Subjects per Split")
    plt.ylabel("# Subjects")
    for bar, c in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, str(c),
                 ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    if out_png:
        plt.savefig(out_png, dpi=200, bbox_inches='tight')
        print(f"[SAVED] {out_png}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize which subjects went to which split")
    #parser.add_argument("--split_root", required=True, help="Path to split root (contains train/val/test)")
    parser.add_argument("--pivot_csv", default="split_subjects_pivot.csv", help="Output CSV with train/val/test columns and counts")
    parser.add_argument("--case_csv", default="split_subjects_cases.csv", help="Output CSV with case-variant details and same-split flag")
    parser.add_argument("--csv", default="split_subjects.csv", help="Output CSV path")
    parser.add_argument("--png", default="split_subjects.png", help="Output PNG path for bar chart (optional)")
    args = parser.parse_args()

    #mapping = collect_subjects(args.split_root)
    #case_counts = collect_subjects_with_cases(args.split_root)
    mapping = collect_subjects("splitted_dataset")
    case_counts = collect_subjects_with_cases("splitted_dataset")
    print_overview(mapping)
    export_csv(mapping, args.csv)
    export_pivot_csv(mapping, case_counts, args.pivot_csv)
    export_case_csv(case_counts, args.case_csv)
    # Simple graphical summary
    plot_counts(mapping, args.png)


if __name__ == "__main__":
    main()


