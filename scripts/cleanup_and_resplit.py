import os
import shutil
import stat
import argparse

from src.split_person_wise import split_person_wise_unified


def _on_rm_error(func, path, exc_info):
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except Exception:
        pass


def clean_and_resplit(raw: str, out: str, seed: int = 42,
                      train: float = 0.7, val: float = 0.15, test: float = 0.15,
                      classes=("NotDrowsy", "Drowsy")):
    ratios = {"train": train, "val": val, "test": test}
    if abs(sum(ratios.values()) - 1.0) > 1e-6:
        raise ValueError("Ratios must sum to 1.0")

    if os.path.exists(out):
        print(f"[INFO] Removing existing directory: {out}")
        shutil.rmtree(out, onerror=_on_rm_error)

    print("[INFO] Rebuilding split...")
    summary = split_person_wise_unified(
        raw_directory=raw,
        output_directory=out,
        classes=classes,
        split_ratios=ratios,
        seed=seed,
        force=False,
    )

    print("=== Split summary ===")
    for split in ("train", "val", "test"):
        parts = ", ".join(f"{cls}:{summary[split][cls]}" for cls in summary[split])
        print(f"{split:5s} -> {parts}")
    print(f"[DONE] Output written to: {out}")


def main():
    parser = argparse.ArgumentParser(description="Clean output dir and run unified person-wise split")
    parser.add_argument("--raw", required=True, help="Path to raw dataset root (contains class folders)")
    parser.add_argument("--out", required=True, help="Path to output split root to recreate")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train", type=float, default=0.7)
    parser.add_argument("--val", type=float, default=0.15)
    parser.add_argument("--test", type=float, default=0.15)
    args = parser.parse_args()

    clean_and_resplit(
        raw=args.raw,
        out=args.out,
        seed=args.seed,
        train=args.train,
        val=args.val,
        test=args.test,
    )


if __name__ == "__main__":
    main()


