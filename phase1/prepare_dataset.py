"""
Unify Stanford Cars + VMMRdb (+ optionally CompCars) into a common format,
strip years from labels to get make/model classes, canonicalize label spelling
across datasets, and produce train/val/test CSV splits.

Usage:
    python prepare_dataset.py
    python prepare_dataset.py --min-count 8   # stricter long-tail cutoff

All source labels are normalized via normalize_labels.canonical_label() so the same
physical car from different datasets collapses to one class.
"""
import argparse
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from normalize_labels import canonical_label

DATA_DIR   = Path(__file__).parent.parent / "data"
RAW_DIR    = DATA_DIR / "raw"
SPLITS_DIR = DATA_DIR / "splits"

_YEAR_RE = re.compile(r"\s+\d{4}$")


def process_stanford_cars():
    stanford_raw = RAW_DIR / "stanford_cars"
    if not stanford_raw.exists():
        print("Stanford Cars not found at data/raw/stanford_cars — run download_data.py first.")
        return []

    records = []
    for split in ["train", "test"]:
        split_dir = stanford_raw / split
        if not split_dir.exists():
            continue
        for class_dir in sorted(split_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            label = canonical_label(class_dir.name)  # strips year, unifies spelling; keeps body style
            if not label:
                continue
            for img_path in class_dir.glob("*.jpg"):
                records.append({
                    "path": str(img_path),
                    "label": label,
                    "source": "stanford",
                    "original_split": split,
                })

    n_classes = len({r["label"] for r in records})
    print(f"Stanford Cars: {len(records):,} images across {n_classes} make/model classes")
    return records


def process_vmmrdb():
    """VMMRdb: one folder per class named '{make}_{model}_{year}', e.g. 'honda_accord_2003'."""
    vmmr_raw = RAW_DIR / "vmmrdb"
    if not vmmr_raw.exists():
        print("VMMRdb not found at data/raw/vmmrdb — skipping (run download_data.py).")
        return []

    records = []
    skipped = 0
    for class_dir in sorted(vmmr_raw.iterdir()):
        if not class_dir.is_dir():
            continue
        label = canonical_label(class_dir.name)  # 'honda_accord_2003' -> 'Honda Accord'
        if not label:
            skipped += 1
            continue
        imgs = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.jpeg")) + list(class_dir.glob("*.png"))
        for img_path in imgs:
            records.append({
                "path": str(img_path),
                "label": label,
                "source": "vmmrdb",
                "original_split": "train",  # VMMRdb has no predefined split; carved in create_splits
            })

    n_classes = len({r["label"] for r in records})
    print(f"VMMRdb: {len(records):,} images across {n_classes} make/model classes "
          f"({skipped} folders skipped)")
    return records


def process_comp_cars():
    comp_raw = RAW_DIR / "comp_cars"
    if not comp_raw.exists():
        print("CompCars not found — skipping (optional dataset).")
        return []

    make_file  = comp_raw / "label" / "make_names.txt"
    model_file = comp_raw / "label" / "model_names.txt"
    if not make_file.exists() or not model_file.exists():
        print("CompCars label files missing — skipping.")
        return []

    makes = {}
    with open(make_file) as f:
        for i, line in enumerate(f, 1):
            makes[i] = line.strip()

    models = {}
    with open(model_file) as f:
        for i, line in enumerate(f, 1):
            parts = line.strip().split()
            models[i] = (int(parts[0]), " ".join(parts[1:]))

    records = []
    for make_dir in (comp_raw / "image").iterdir():
        if not make_dir.is_dir():
            continue
        try:
            make_id = int(make_dir.name)
        except ValueError:
            continue
        make_name = makes.get(make_id, make_dir.name)
        for model_dir in make_dir.iterdir():
            if not model_dir.is_dir():
                continue
            try:
                model_id = int(model_dir.name)
            except ValueError:
                continue
            _, model_name = models.get(model_id, (None, model_dir.name))
            label = canonical_label(f"{make_name} {model_name}")
            if not label:
                continue
            for year_dir in model_dir.iterdir():
                for img_path in year_dir.glob("*.jpg"):
                    records.append({
                        "path": str(img_path),
                        "label": label,
                        "source": "compcars",
                        "original_split": "train",
                    })

    n_classes = len({r["label"] for r in records})
    print(f"CompCars: {len(records):,} images across {n_classes} make/model classes")
    return records


def create_splits(records: list, val_fraction: float = 0.1, test_fraction: float = 0.1,
                  min_count: int = 5, seed: int = 42):
    df = pd.DataFrame(records)

    # Drop classes with too few examples to stratify a train/val/test split.
    counts = df["label"].value_counts()
    valid_classes = counts[counts >= min_count].index
    df = df[df["label"].isin(valid_classes)].copy()
    print(f"\nAfter min-count>={min_count} filter: {df['label'].nunique()} classes, {len(df):,} images")

    # Stanford's own test folder stays the isolated benchmark; everything else is pooled.
    stanford_test = df[(df["source"] == "stanford") & (df["original_split"] == "test")]
    pool          = df[~df.index.isin(stanford_test.index)].copy()

    # Carve a stratified per-source test split from the pool (mainly VMMRdb/CompCars).
    # Only labels with >=2 pooled samples can be stratified into a test slice.
    pool_counts = pool["label"].value_counts()
    strat_labels = pool_counts[pool_counts >= 2].index
    pool_strat   = pool[pool["label"].isin(strat_labels)]
    pool_rest    = pool[~pool["label"].isin(strat_labels)]

    pool_trainval, pool_test = train_test_split(
        pool_strat, test_size=test_fraction, stratify=pool_strat["label"], random_state=seed,
    )
    pool_trainval = pd.concat([pool_trainval, pool_rest])  # tiny classes go entirely to train/val

    # train/val split of the combined train-val pool
    train_df, val_df = train_test_split(
        pool_trainval, test_size=val_fraction, stratify=pool_trainval["label"], random_state=seed,
    )

    test_df = pd.concat([stanford_test, pool_test])

    all_classes = sorted(df["label"].unique().tolist())
    label2idx   = {c: i for i, c in enumerate(all_classes)}
    for split_df in (train_df, val_df, test_df):
        split_df["label_idx"] = split_df["label"].map(label2idx)

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(SPLITS_DIR / "train.csv", index=False)
    val_df.to_csv(  SPLITS_DIR / "val.csv",   index=False)
    test_df.to_csv( SPLITS_DIR / "test.csv",  index=False)
    # Per-source test CSVs for A7 evaluation (Stanford benchmark vs VMMRdb in-the-wild).
    test_df[test_df["source"] == "stanford"].to_csv(SPLITS_DIR / "test_stanford.csv", index=False)
    test_df[test_df["source"] == "vmmrdb"].to_csv(  SPLITS_DIR / "test_vmmrdb.csv",   index=False)
    pd.Series(all_classes).to_csv(SPLITS_DIR / "classes.txt", index=False, header=False)

    print(f"\nSplits written to {SPLITS_DIR}")
    print(f"  Train : {len(train_df):,} images")
    print(f"  Val   : {len(val_df):,} images")
    print(f"  Test  : {len(test_df):,} images "
          f"(stanford {int((test_df['source']=='stanford').sum()):,} / "
          f"vmmrdb {int((test_df['source']=='vmmrdb').sum()):,})")
    print(f"  Classes: {len(all_classes)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-count", type=int, default=5,
                        help="Drop classes with fewer than this many images")
    args = parser.parse_args()

    records = process_stanford_cars() + process_vmmrdb() + process_comp_cars()
    if not records:
        print("No data found. Run download_data.py first.")
    else:
        create_splits(records, min_count=args.min_count)
