"""
Download Stanford Cars dataset from HuggingFace.
CompCars requires manual registration -- instructions are printed at the end.

Usage:
    python download_data.py
    python download_data.py --skip-stanford   # skip if already downloaded
"""
import argparse
from pathlib import Path
from tqdm import tqdm

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"

# Characters invalid in Windows directory names
_INVALID_CHARS = r'\/:*?"<>|'


def sanitize_dirname(name: str) -> str:
    """Replace Windows-invalid path characters so labels can be used as folder names."""
    for ch in _INVALID_CHARS:
        name = name.replace(ch, "-")
    return name.strip()


def download_stanford_cars():
    from datasets import load_dataset

    out_dir = DATA_DIR / "stanford_cars"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading Stanford Cars from HuggingFace (tanganke/stanford_cars)...")
    print("First run will download ~1.8 GB — subsequent runs use the HF cache.\n")
    ds = load_dataset("tanganke/stanford_cars")

    for split_name, split in ds.items():
        split_dir = out_dir / split_name
        split_dir.mkdir(exist_ok=True)
        label_names = split.features["label"].names
        print(f"Saving {split_name} split ({len(split)} images)...")
        for i, example in enumerate(tqdm(split)):
            label = sanitize_dirname(label_names[example["label"]])
            label_dir = split_dir / label
            label_dir.mkdir(exist_ok=True)
            img_path = label_dir / f"{i:05d}.jpg"
            if not img_path.exists():
                example["image"].save(img_path)

    num_classes = len(ds["train"].features["label"].names)
    print(f"\nStanford Cars saved to {out_dir}")
    print(f"196 year/make/model classes -> will merge to make/model in prepare_dataset.py")


def download_vmmrdb():
    """
    Download + extract VMMRdb (Vehicle Make and Model Recognition Dataset).
    ~12.3 GB zip, 291,752 real-world (Craigslist) images, 9,170 make/model/year classes,
    years 1950-2016, MIT-licensed. Folder-per-class layout: '{make}_{model}_{year}/'.
    """
    import zipfile
    import urllib.request

    out_dir = DATA_DIR / "vmmrdb"
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / "VMMRdb.zip"
    url = "https://www.dropbox.com/s/uwa7c5uz7cac7cw/VMMRdb.zip?dl=1"

    if not zip_path.exists():
        print(f"Downloading VMMRdb (~12.3 GB) to {zip_path} ...")
        urllib.request.urlretrieve(url, zip_path)
    else:
        print(f"VMMRdb.zip already present ({zip_path.stat().st_size:,} bytes) — skipping download.")

    print("Extracting VMMRdb.zip (294k files, this takes a few minutes) ...")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(out_dir)

    n = sum(1 for p in out_dir.iterdir() if p.is_dir())
    print(f"VMMRdb ready at {out_dir}  ({n} class folders)")


def print_compcars_instructions():
    print()
    print("=" * 60)
    print("CompCars Dataset (manual download required)")
    print("=" * 60)
    print("1. Go to: http://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/")
    print("2. Request access and download the web-nature subset")
    print("3. Extract to:  data/raw/comp_cars/")
    print()
    print("   Expected structure after extracting:")
    print("   data/raw/comp_cars/")
    print("     image/<make_id>/<model_id>/<year>/<img>.jpg")
    print("     label/make_names.txt")
    print("     label/model_names.txt")
    print()
    print("CompCars adds ~30k images and 1,700+ model classes,")
    print("especially helpful for non-US and post-2012 vehicles.")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-stanford", action="store_true",
                        help="Skip Stanford Cars download (already done)")
    parser.add_argument("--vmmrdb", action="store_true",
                        help="Download + extract the VMMRdb dataset")
    args = parser.parse_args()

    if not args.skip_stanford:
        download_stanford_cars()
    if args.vmmrdb:
        download_vmmrdb()
    print_compcars_instructions()
