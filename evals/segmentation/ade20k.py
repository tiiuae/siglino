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

from utils import (
    build_backbone_and_processor,
    make_collate_fn,
    AMOELinearSeg,
    FEATURE_DIM_DICT,
)

# --------------------
# Datasets
# --------------------

class ADE20KDataset(Dataset):
    """
    ADE20K semantic segmentation dataset with structure:
      root/
        images/{split}/*.jpg|*.png
        annotations/{split}/*.png
    Mask values: 0 is ignore, 1..K are classes. Resized to (IMAGE_SIZE, IMAGE_SIZE).
    """
    def __init__(self, root_dir, split='training', image_size=256):
        self.root_dir = root_dir
        self.split = split
        self.image_dir = os.path.join(root_dir, 'images', split)
        self.annotation_dir = os.path.join(root_dir, 'annotations', split)

        assert os.path.isdir(self.image_dir), f"Missing images dir: {self.image_dir}"
        assert os.path.isdir(self.annotation_dir), f"Missing annotations dir: {self.annotation_dir}"

        self.images = sorted([os.path.join(self.image_dir, f) for f in os.listdir(self.image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        self.masks = sorted([os.path.join(self.annotation_dir, f) for f in os.listdir(self.annotation_dir) if f.lower().endswith('.png')])
        assert len(self.images) == len(self.masks), f"Images and masks count mismatch: {len(self.images)} vs {len(self.masks)}"

        self.resize_img = transforms.Resize((image_size, image_size), interpolation=Image.BICUBIC)
        self.resize_mask = transforms.Resize((image_size, image_size), interpolation=Image.NEAREST)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('RGB')
        msk = Image.open(self.masks[idx])
        img = self.resize_img(img)
        msk = self.resize_mask(msk)
        msk = torch.from_numpy(np.array(msk)).long()  # 0..K (0 = ignore)
        return img, msk  # PIL, LongTensor(H,W)


def get_dataset(root_dir: str, image_size: int):
    split_train = 'training'
    split_val = 'validation'
    train_ds = ADE20KDataset(root_dir, split=split_train, image_size=image_size)
    val_ds = ADE20KDataset(root_dir, split=split_val, image_size=image_size)
    return train_ds, val_ds

# --------------------
# Evaluation
# --------------------

@torch.no_grad()
def evaluate_seg(model, dataloader, criterion, num_classes: int, device='cuda'):
    model.eval()
    val_loss = 0.0
    k = num_classes - 1  # classes excluding ignore 0
    conf = np.zeros((k, k), dtype=np.int64)

    for batch in dataloader:
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        spatial_shape = batch["spatial_shape"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True)  # Long

        logits = model(pixel_values, spatial_shape)
        loss = criterion(logits, targets)
        val_loss += loss.item()

        preds = logits.argmax(dim=1)  # (B,H,W)
        t = targets
        p = preds
        valid_t = (t > 0) & (t <= k)
        if valid_t.any():
            t = t[valid_t]
            p = p[valid_t]
            valid_p = (p > 0) & (p <= k)
            if valid_p.any():
                t = t[valid_p] - 1
                p = p[valid_p] - 1
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

# --------------------
# Args
# --------------------

def parse_args():
    p = argparse.ArgumentParser("ADE20K segmentation evaluation using Falcon-Omni (distilled, multi-teacher packing) backbone")
    p.add_argument("--root_dir", type=str, required=True, help="Root directory of ADE20K dataset")
    p.add_argument("--ckpt_path", type=str, required=True, help="Path to Falcon-Omni distilled checkpoint (training output checkpoint)")
    p.add_argument("--feature_type", type=str, default="dinov3", choices=["dinov3", "amoe", "siglip2"])
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--num_classes", type=int, default=151, help="For segmentation (includes ignore=0)")
    p.add_argument("--out_dir", type=str, default=None, help="Directory to write JSON metrics; defaults to derived training output dir")
    p.add_argument("--image_size", type=int, default=256, help="Image size")
    return p.parse_args()

# --------------------
# Main
# --------------------

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset_name = "ade20k"
    task = "segmentation"

    image_size = args.image_size
    train_dataset, val_dataset = get_dataset(args.root_dir, image_size=image_size)

    # Backbone + processor
    backbone, image_processor = build_backbone_and_processor(
        ckpt_path=args.ckpt_path,
        device=device,
        feature_type=args.feature_type,
    )

    # Dataloaders
    collate = make_collate_fn(image_processor)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, collate_fn=collate
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, collate_fn=collate
    )

    # Feature dims
    feature_dim = FEATURE_DIM_DICT[args.feature_type]

    # Model, loss, optimizer
    model = AMOELinearSeg(
        backbone, num_classes=args.num_classes,
        feature_dim=feature_dim, feature_type=args.feature_type,
        image_size=image_size
    ).to(device).to(dtype=torch.bfloat16)

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.AdamW(model.head.parameters(), lr=args.lr, weight_decay=1e-3)

    # Train/eval
    best_metric = None
    best_epoch = -1
    metric_name = 'mIoU'

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            spatial_shape = batch["spatial_shape"].to(device, non_blocking=True)
            targets = batch["targets"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(pixel_values, spatial_shape)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= max(1, len(train_loader))
        val_loss, metric = evaluate_seg(model, val_loader, criterion, num_classes=args.num_classes, device=device)
        is_better = (best_metric is None) or (metric > best_metric)  # Higher is better

        if is_better:
            best_metric = metric
            best_epoch = epoch + 1

        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | {metric_name}: {metric:.4f} | Best({metric_name}): {best_metric:.4f} @ epoch {best_epoch}")

    out_dir = args.out_dir or "evals"
    os.makedirs(out_dir, exist_ok=True)

    result = {
        "dataset": dataset_name,
        "task": task,
        "feature_type": args.feature_type,
        "metric_name": metric_name,
        "best_metric_value": float(best_metric) if best_metric is not None else None,
        "best_epoch": best_epoch,
        "ckpt_path": args.ckpt_path,
        "num_classes": args.num_classes,
    }
    json_path = os.path.join(out_dir, f"{dataset_name}_{task}_{args.feature_type}.json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote results to {json_path}")
    print(f"BEST_{metric_name.upper()}: {best_metric:.4f} @ epoch {best_epoch}")

if __name__ == "__main__":
    main()
