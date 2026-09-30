"""
Post-hoc body-style class merging evaluation, with optional TTA.
Aggregates softmax probabilities across body-style variants (Coupe/Convertible/
Sedan/Van/etc.) without retraining. Uses the existing checkpoint as-is.

Usage:
    python evaluate_merged.py            # standard eval with merging
    python evaluate_merged.py --tta      # + TenCrop TTA (10 augmentations/image)
"""
import re
import argparse
from pathlib import Path

import torch
import torchvision.transforms as T
import pandas as pd
from torch.utils.data import DataLoader
from torch.amp import autocast
from tqdm import tqdm
import timm

from dataset import CarDataset, get_transforms, _MEAN, _STD

SPLITS_DIR = Path(__file__).parent.parent / "data" / "splits"
CKPT_DIR   = Path(__file__).parent.parent / "checkpoints"

# Multi-word suffixes must be checked before single-word ones
_BODY_MULTI = [
    "Hybrid Crew Cab", "SuperCrew Cab", "Cargo Van",
    "Crew Cab", "Extended Cab", "Regular Cab", "Club Cab", "Hybrid SUV",
]
_BODY_SINGLE = [
    "Convertible", "Hatchback", "Cabriolet", "Minivan",
    "Pickup", "Coupe", "Sedan", "Wagon", "SUV", "Van",
]


class StackCrops:
    """Applies ToTensor + Normalize to each crop and stacks into (N, C, H, W)."""
    def __init__(self):
        self.normalize = T.Compose([T.ToTensor(), T.Normalize(_MEAN, _STD)])

    def __call__(self, crops):
        return torch.stack([self.normalize(c) for c in crops])


def get_tta_transform(img_size: int = 384) -> T.Compose:
    """TenCrop: center + 4 corners, each × horizontal flip = 10 views per image."""
    return T.Compose([
        T.Resize(int(img_size * 1.14)),
        T.TenCrop(img_size),
        StackCrops(),  # (10, C, H, W)
    ])


def strip_body_style(label: str) -> str:
    for bs in _BODY_MULTI + _BODY_SINGLE:
        label = re.sub(rf"\s+{re.escape(bs)}$", "", label, flags=re.IGNORECASE).strip()
    return label


def build_merge_mapping(classes: list):
    """
    Returns:
      merged_classes  : sorted list of unique merged class names
      old_to_merged   : LongTensor (num_old_classes,) — old idx -> merged idx
      merge_groups    : dict merged_name -> [old_name, ...] for reporting
    """
    merged_names = [strip_body_style(c) for c in classes]
    unique_merged = sorted(set(merged_names))
    name_to_idx   = {n: i for i, n in enumerate(unique_merged)}
    old_to_merged = torch.tensor([name_to_idx[n] for n in merged_names], dtype=torch.long)

    merge_groups: dict = {}
    for old_name, merged_name in zip(classes, merged_names):
        merge_groups.setdefault(merged_name, []).append(old_name)

    return unique_merged, old_to_merged, merge_groups


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt        = torch.load(args.checkpoint, map_location=device)
    saved_args  = ckpt["args"]
    num_classes = ckpt["num_classes"]
    img_size    = saved_args.get("img_size", 384)
    model_name  = saved_args.get("model", "efficientnet_b4")
    print(f"Checkpoint: epoch {ckpt['epoch']}, val top-1 {ckpt['val_top1']:.1f}%")

    old_classes = pd.read_csv(SPLITS_DIR / "classes.txt", header=None)[0].tolist()
    assert len(old_classes) == num_classes

    merged_classes, old_to_merged, merge_groups = build_merge_mapping(old_classes)
    num_merged  = len(merged_classes)
    num_removed = num_classes - num_merged

    print(f"\nClass merging summary:")
    print(f"  Original : {num_classes} classes")
    print(f"  Merged   : {num_merged} classes  (-{num_removed} body-style variants)")

    actually_merged = {k: v for k, v in merge_groups.items() if len(v) > 1}
    if actually_merged:
        print(f"\n  Variants that were merged ({len(actually_merged)} groups):")
        for name, variants in sorted(actually_merged.items()):
            print(f"    {name}  <-  {', '.join(variants)}")

    old_to_merged_dev = old_to_merged.to(device)

    transform = get_tta_transform(img_size) if args.tta else get_transforms(train=False, img_size=img_size)
    test_ds   = CarDataset(args.split, transform)
    print(f"Eval split: {args.split}  ({len(test_ds):,} images)")
    # TTA returns (10, C, H, W) per image so we reduce batch size to avoid OOM
    bs = max(1, args.batch_size // 10) if args.tta else args.batch_size
    loader = DataLoader(test_ds, batch_size=bs, num_workers=4, pin_memory=True)

    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device).eval()

    mode_str = "TTA (10 crops)" if args.tta else "standard"
    print(f"Inference mode: {mode_str}")

    all_preds, all_labels = [], []

    with torch.no_grad():
        for imgs, old_labels in tqdm(loader, desc="evaluating"):
            old_labels = old_labels.to(device)

            if args.tta:
                # imgs: (B, 10, C, H, W) — run all crops through model, average probs
                B, n_crops, C, H, W = imgs.shape
                imgs_flat = imgs.view(B * n_crops, C, H, W).to(device)
                with autocast("cuda"):
                    logits = model(imgs_flat)                              # (B*10, num_old)
                probs = torch.softmax(logits.float(), dim=1)
                probs = probs.view(B, n_crops, -1).mean(dim=1)            # (B, num_old)
            else:
                imgs = imgs.to(device)
                with autocast("cuda"):
                    logits = model(imgs)
                probs = torch.softmax(logits.float(), dim=1)               # (B, num_old)

            B = probs.size(0)
            merged_probs = torch.zeros(B, num_merged, device=device)
            idx_expanded = old_to_merged_dev.unsqueeze(0).expand(B, -1)
            merged_probs.scatter_add_(1, idx_expanded, probs)

            all_preds.append(merged_probs.argmax(1).cpu())
            all_labels.append(old_to_merged_dev[old_labels].cpu())

    preds  = torch.cat(all_preds)
    labels = torch.cat(all_labels)

    top1 = (preds == labels).float().mean().item() * 100
    gate = "PASSED" if top1 >= 85.0 else "FAILED"

    print(f"\n{'='*42}")
    print(f"Test set  ({len(labels):,} images, {num_merged} merged classes)")
    print(f"  Top-1 accuracy : {top1:.2f}%")
    print(f"  Phase 1 gate   : {gate}")
    print(f"{'='*42}")

    # Per-class breakdown
    labels_np = labels.numpy()
    preds_np  = preds.numpy()
    per_class = {}
    for i, name in enumerate(merged_classes):
        mask = labels_np == i
        if mask.sum() == 0:
            continue
        per_class[name] = (preds_np[mask] == i).mean() * 100

    worst = sorted(per_class.items(), key=lambda x: x[1])[:10]
    print("\nBottom 10 merged classes:")
    for name, acc in worst:
        print(f"  {acc:5.1f}%  {name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(CKPT_DIR / "efficientnet_b4_best.pth"))
    parser.add_argument("--split", default="test",
                        help="Which split CSV to evaluate (test, test_stanford, test_vmmrdb)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--tta", action="store_true", help="Enable TenCrop TTA")
    main(parser.parse_args())
