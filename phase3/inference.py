"""
Phase 1 model -> Phase 3 inference.

Loads the EfficientNet-B4 checkpoint and reproduces the body-style class merging
from phase1/evaluate_merged.py (189 raw classes -> 174 merged make/model classes)
so predictions come back as "Make Model" strings the catalog can look up.
"""
import os
import re
from functools import lru_cache
from pathlib import Path

import timm
import torch
import torchvision.transforms as T
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SPLITS_DIR = ROOT / "data" / "splits"
# VMMRdb-expanded model (1,173 classes, trained 2026-07-04). Override with CARSPOTTERS_CKPT.
# The original Stanford-only checkpoint is efficientnet_b4_best.pth (preserved).
CKPT_PATH = Path(os.getenv("CARSPOTTERS_CKPT",
                           str(ROOT / "checkpoints" / "efficientnet_b4_vmmrdb_best.pth")))

_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]

# Same suffix lists as phase1/evaluate_merged.py — multi-word first.
_BODY_MULTI = [
    "Hybrid Crew Cab", "SuperCrew Cab", "Cargo Van",
    "Crew Cab", "Extended Cab", "Regular Cab", "Club Cab", "Hybrid SUV",
]
_BODY_SINGLE = [
    "Convertible", "Hatchback", "Cabriolet", "Minivan",
    "Pickup", "Coupe", "Sedan", "Wagon", "SUV", "Van",
]


def strip_body_style(label: str) -> str:
    for bs in _BODY_MULTI + _BODY_SINGLE:
        label = re.sub(rf"\s+{re.escape(bs)}$", "", label, flags=re.IGNORECASE).strip()
    return label


def _build_merge_mapping(classes: list[str]):
    merged_names = [strip_body_style(c) for c in classes]
    unique_merged = sorted(set(merged_names))
    name_to_idx = {n: i for i, n in enumerate(unique_merged)}
    old_to_merged = torch.tensor(
        [name_to_idx[n] for n in merged_names], dtype=torch.long
    )
    return unique_merged, old_to_merged


class Predictor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(CKPT_PATH, map_location=self.device)
        saved_args = ckpt["args"]
        self.num_classes = ckpt["num_classes"]
        self.img_size = saved_args.get("img_size", 384)
        model_name = saved_args.get("model", "efficientnet_b4")

        classes = [
            line.strip()
            for line in (SPLITS_DIR / "classes.txt").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.merged_classes, old_to_merged = _build_merge_mapping(classes)
        self.old_to_merged = old_to_merged.to(self.device)
        self.num_merged = len(self.merged_classes)

        self.model = timm.create_model(
            model_name, pretrained=False, num_classes=self.num_classes
        )
        self.model.load_state_dict(ckpt["model_state"])
        self.model.to(self.device).eval()

        self.transform = T.Compose([
            T.Resize(int(self.img_size * 1.14)),
            T.CenterCrop(self.img_size),
            T.ToTensor(),
            T.Normalize(_MEAN, _STD),
        ])

    @torch.no_grad()
    def predict(self, img: Image.Image, topk: int = 3) -> list[tuple[str, float]]:
        x = self.transform(img.convert("RGB")).unsqueeze(0).to(self.device)
        logits = self.model(x)
        probs = torch.softmax(logits.float(), dim=1)                 # (1, num_old)
        merged = torch.zeros(1, self.num_merged, device=self.device)
        merged.scatter_add_(1, self.old_to_merged.unsqueeze(0), probs)
        merged = merged[0]
        top_p, top_i = merged.topk(min(topk, self.num_merged))
        return [
            (self.merged_classes[i], float(p))
            for p, i in zip(top_p.tolist(), top_i.tolist())
        ]


@lru_cache(maxsize=1)
def get_predictor() -> Predictor:
    """Lazily load the model once, on first /identify call."""
    return Predictor()
