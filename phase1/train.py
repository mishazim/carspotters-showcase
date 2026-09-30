"""
Train EfficientNet-B4 (or any timm model) on the car make/model dataset.

Usage:
    # Run from phase1/ directory
    python train.py
    python train.py --model efficientnet_b4 --epochs 30 --batch-size 32
    python train.py --model vit_base_patch16_384 --img-size 384 --lr 5e-5
"""
import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
import timm

from dataset import CarDataset, get_transforms

CKPT_DIR = Path(__file__).parent.parent / "checkpoints"


def topk_accuracy(logits: torch.Tensor, targets: torch.Tensor, k: int) -> float:
    with torch.no_grad():
        k = min(k, logits.size(1))
        _, pred = logits.topk(k, dim=1)
        correct = pred.eq(targets.unsqueeze(1).expand_as(pred)).any(dim=1)
        return correct.float().mean().item() * 100


def train_epoch(model, loader, optimizer, criterion, scaler, device):
    model.train()
    total_loss = total_top1 = 0
    for imgs, labels in tqdm(loader, leave=False, desc="train"):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        with autocast("cuda"):
            logits = model(imgs)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        with torch.no_grad():
            total_loss  += loss.item()
            total_top1  += topk_accuracy(logits, labels, k=1)  # reuse logits (no 2nd forward)
    n = len(loader)
    return total_loss / n, total_top1 / n


@torch.no_grad()
def val_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = total_top1 = total_top5 = 0
    for imgs, labels in tqdm(loader, leave=False, desc="val"):
        imgs, labels = imgs.to(device), labels.to(device)
        with autocast("cuda"):
            logits = model(imgs)
            total_loss += criterion(logits, labels).item()
        total_top1 += topk_accuracy(logits, labels, k=1)
        total_top5 += topk_accuracy(logits, labels, k=5)
    n = len(loader)
    return total_loss / n, total_top1 / n, total_top5 / n


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cpu":
        print("Warning: training on CPU will be very slow. Connect a GPU.")

    train_ds = CarDataset("train", get_transforms(train=True,  img_size=args.img_size))
    val_ds   = CarDataset("val",   get_transforms(train=False, img_size=args.img_size))
    num_classes = train_ds.num_classes
    print(f"Classes: {num_classes}  |  Train: {len(train_ds):,}  Val: {len(val_ds):,}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size * 2, shuffle=False,
        num_workers=args.workers, pin_memory=True, persistent_workers=True,
    )

    model = timm.create_model(args.model, pretrained=True, num_classes=num_classes)
    model = model.to(device)
    print(f"Model: {args.model}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler    = GradScaler("cuda")

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    best_top1 = 0.0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_top1        = train_epoch(model, train_loader, optimizer, criterion, scaler, device)
        va_loss, va_top1, va_top5 = val_epoch(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train loss {tr_loss:.4f} top1 {tr_top1:.1f}% | "
            f"val loss {va_loss:.4f} top1 {va_top1:.1f}% top5 {va_top5:.1f}% | "
            f"{elapsed:.0f}s"
        )

        if va_top1 > best_top1:
            best_top1 = va_top1
            suffix = f"_{args.tag}" if args.tag else ""
            ckpt = CKPT_DIR / f"{args.model}{suffix}_best.pth"
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_top1": va_top1,
                "num_classes": num_classes,
                "args": vars(args),
            }, ckpt)
            print(f"  -> Saved best checkpoint  ({va_top1:.1f}%)")

    print(f"\nBest val top-1: {best_top1:.1f}%  (gate: 85.0%)")
    if best_top1 >= 85.0:
        print("Gate PASSED — ready to proceed to Phase 2.")
    else:
        print("Gate NOT MET — consider more data, longer training, or ViT backbone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      default="efficientnet_b4")
    parser.add_argument("--tag",        default="",
                        help="Checkpoint name suffix (e.g. 'v2') so runs don't overwrite each other")
    parser.add_argument("--img-size",   type=int,   default=384)
    parser.add_argument("--batch-size", type=int,   default=32)
    parser.add_argument("--epochs",     type=int,   default=30)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--workers",    type=int,   default=4)
    main(parser.parse_args())
