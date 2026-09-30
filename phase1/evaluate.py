"""
Run the Phase 1 gate evaluation on the held-out test set.
Prints top-1/top-5 accuracy and the 10 hardest classes.

Usage:
    python evaluate.py
    python evaluate.py --checkpoint ../checkpoints/efficientnet_b4_best.pth
"""
import argparse
from pathlib import Path

import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torch.amp import autocast
from tqdm import tqdm
import timm

from dataset import CarDataset, get_transforms

SPLITS_DIR = Path(__file__).parent.parent / "data" / "splits"
CKPT_DIR   = Path(__file__).parent.parent / "checkpoints"


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location=device)
    saved_args  = ckpt["args"]
    num_classes = ckpt["num_classes"]
    img_size    = saved_args.get("img_size", 384)
    model_name  = saved_args.get("model", "efficientnet_b4")
    print(f"Checkpoint: epoch {ckpt['epoch']}, val top-1 {ckpt['val_top1']:.1f}%")

    test_ds = CarDataset("test", get_transforms(train=False, img_size=img_size))
    loader  = DataLoader(test_ds, batch_size=args.batch_size,
                         num_workers=4, pin_memory=True)

    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device).eval()

    classes = pd.read_csv(SPLITS_DIR / "classes.txt", header=None)[0].tolist()

    all_logits, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="evaluating"):
            imgs = imgs.to(device)
            with autocast("cuda"):
                all_logits.append(model(imgs).cpu())
            all_labels.append(labels)

    logits = torch.cat(all_logits)   # (N, C)
    labels = torch.cat(all_labels)   # (N,)
    preds  = logits.argmax(1)

    top1 = (preds == labels).float().mean().item() * 100
    k    = min(5, logits.size(1))
    top5 = labels.unsqueeze(1).eq(logits.topk(k, dim=1).indices).any(1).float().mean().item() * 100

    print(f"\n{'='*40}")
    print(f"Test set  ({len(labels):,} images, {num_classes} classes)")
    print(f"  Top-1: {top1:.2f}%")
    print(f"  Top-5: {top5:.2f}%")
    gate = "PASSED" if top1 >= 85.0 else "FAILED"
    print(f"  Phase 1 gate (85% top-1): {gate}")
    print(f"{'='*40}")

    # Per-class breakdown
    labels_np = labels.numpy()
    preds_np  = preds.numpy()
    per_class_acc = {}
    for i, name in enumerate(classes):
        mask = labels_np == i
        if mask.sum() == 0:
            continue
        per_class_acc[name] = (preds_np[mask] == i).mean() * 100

    worst = sorted(per_class_acc.items(), key=lambda x: x[1])[:10]
    print("\nBottom 10 classes (hardest):")
    for name, acc in worst:
        print(f"  {acc:5.1f}%  {name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default=str(CKPT_DIR / "efficientnet_b4_best.pth"),
    )
    parser.add_argument("--batch-size", type=int, default=64)
    main(parser.parse_args())
