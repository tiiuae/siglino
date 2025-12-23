import os
import argparse
import json
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

from datasets import load_dataset

from utils import (
    FEATURE_DIM_DICT,
    build_backbone_and_processor,
    make_collate_fn,
    AMOELinearSeg,
)

NUM_CLASSES = 21
IGNORE_INDEX = 255

VOC_COLORS = np.array([
    [  0,   0,   0],  # 0 background
    [128,   0,   0],  # 1 aeroplane
    [  0, 128,   0],  # 2 bicycle
    [128, 128,   0],  # 3 bird
    [  0,   0, 128],  # 4 boat
    [128,   0, 128],  # 5 bottle
    [  0, 128, 128],  # 6 bus
    [128, 128, 128],  # 7 car
    [ 64,   0,   0],  # 8 cat
    [192,   0,   0],  # 9 chair
    [ 64, 128,   0],  # 10 cow
    [192, 128,   0],  # 11 diningtable
    [ 64,   0, 128],  # 12 dog
    [192,   0, 128],  # 13 horse
    [ 64, 128, 128],  # 14 motorbike
    [192, 128, 128],  # 15 person
    [  0,  64,   0],  # 16 pottedplant
    [128,  64,   0],  # 17 sheep
    [  0, 192,   0],  # 18 sofa
    [128, 192,   0],  # 19 train
    [  0,  64, 128],  # 20 tvmonitor
], dtype=np.uint8)

def voc_color_to_index(mask_rgb: np.ndarray) -> np.ndarray:
    h, w, _ = mask_rgb.shape
    out = np.full((h, w), 255, dtype=np.uint8)
    for idx, color in enumerate(VOC_COLORS):
        matches = np.all(mask_rgb == color, axis=-1)
        out[matches] = idx
    return out


class HFPascalVOCDataset(Dataset):
    def __init__(self, split='train', image_size=256, repo_id="nateraw/pascal-voc-2012"):
        self.ds = load_dataset(repo_id, split=split)
        self.image_size = image_size
        self.resize_img = transforms.Resize((image_size, image_size), interpolation=Image.BICUBIC)
        self.resize_mask = transforms.Resize((image_size, image_size), interpolation=Image.NEAREST)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        ex = self.ds[idx]
        img = ex['image']
        msk = ex['mask']
        img = self.resize_img(img.convert('RGB'))
        msk = self.resize_mask(msk)
        arr = np.array(msk, dtype=np.uint8)
        if arr.ndim == 3 and arr.shape[2] == 3:
            arr = voc_color_to_index(arr)
        msk = torch.from_numpy(arr).long()
        return img, msk

@torch.no_grad()
def evaluate_pascal(model, dataloader, criterion, device='cuda'):
    model.eval()
    val_loss = 0.0
    k = NUM_CLASSES
    conf = np.zeros((k, k), dtype=np.int64)

    for batch in dataloader:
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        spatial_shape = batch["spatial_shape"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True)
        logits = model(pixel_values, spatial_shape)
        loss = criterion(logits, targets)
        val_loss += loss.item()

        preds = logits.argmax(dim=1)
        t = targets
        p = preds
        valid = (t != IGNORE_INDEX) & (t >= 0) & (t < k)
        if valid.any():
            t = t[valid]
            p = p[valid].clamp(min=0, max=k-1)
            n = k
            idx = (t * n + p).view(-1)
            binc = torch.bincount(idx, minlength=n * n)
            conf += binc.view(n, n).cpu().numpy()

    inter = np.diag(conf)
    union = conf.sum(1) + conf.sum(0) - inter
    valid = union > 0
    miou = float((inter[valid] / (union[valid] + 1e-10)).mean()) if valid.any() else 0.0
    val_loss = val_loss / max(1, len(dataloader))
    return val_loss, miou


def parse_args():
    p = argparse.ArgumentParser("Pascal VOC 2012 segmentation using Falcon Vision backbone")
    p.add_argument("--ckpt_path", type=str, required=True, help="Path to checkpoint")
    p.add_argument("--feature_type", type=str, default="dinov3", choices=["dinov3", "amoe", "siglip2"])
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--out_dir", type=str, default=None)
    p.add_argument("--hf_repo", type=str, default="nateraw/pascal-voc-2012")
    p.add_argument("--image_size", type=int, default=256)
    return p.parse_args()

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset = HFPascalVOCDataset(split='train', image_size=args.image_size, repo_id=args.hf_repo)
    val_dataset = HFPascalVOCDataset(split='val', image_size=args.image_size, repo_id=args.hf_repo)

    backbone, image_processor = build_backbone_and_processor(
        ckpt_path=args.ckpt_path,
        device=device,
        feature_type=args.feature_type,
    )

    feature_dim = FEATURE_DIM_DICT[args.feature_type]

    model = AMOELinearSeg(
        backbone, num_classes=NUM_CLASSES,
        feature_dim=feature_dim, feature_type=args.feature_type,
        image_size=args.image_size
    ).to(device).to(dtype=torch.bfloat16)
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    optimizer = optim.AdamW(model.head.parameters(), lr=args.lr, weight_decay=1e-3)

    collate = make_collate_fn(image_processor)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, collate_fn=collate
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, collate_fn=collate
    )

    best_miou = None
    best_epoch = -1

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            spatial_shape = batch["spatial_shape"].to(device, non_blocking=True)
            targets = batch["targets"].to(device, non_blocking=True)
            logits = model(pixel_values, spatial_shape)

            optimizer.zero_grad(set_to_none=True)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= max(1, len(train_loader))
        val_loss, miou = evaluate_pascal(model, val_loader, criterion, device=device)
        is_better = (best_miou is None) or (miou > best_miou)
        if is_better:
            best_miou = miou
            best_epoch = epoch + 1

        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | mIoU: {miou:.4f} | Best: {best_miou:.4f} @ epoch {best_epoch}")

    out_dir = args.out_dir or "evals"
    os.makedirs(out_dir, exist_ok=True)

    result = {
        "dataset": "pascal_voc_2012",
        "task": "segmentation",
        "feature_type": args.feature_type,
        "metric_name": "mIoU",
        "best_metric_value": float(best_miou) if best_miou is not None else None,
        "best_epoch": best_epoch,
        "ckpt_path": args.ckpt_path,
        "num_classes": NUM_CLASSES,
        "ignore_index": IGNORE_INDEX,
    }
    json_path = os.path.join(out_dir, f"pascalvoc_segmentation_{args.feature_type}.json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote results to {json_path}")
    print(f"BEST_MIOU: {best_miou:.4f} @ epoch {best_epoch}")

if __name__ == "__main__":
    main()
