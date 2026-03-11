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
    build_backbone_and_processor,
    make_collate_fn,
    AMOELinearSeg,
    FEATURE_DIM_DICT,
)

# Cityscapes labelId -> trainId mapping (unlabeled set to 255)
ID_TO_TRAINID = np.full(256, 255, dtype=np.int64)
ID_TO_TRAINID[7]  = 0   # road
ID_TO_TRAINID[8]  = 1   # sidewalk
ID_TO_TRAINID[11] = 2   # building
ID_TO_TRAINID[12] = 3   # wall
ID_TO_TRAINID[13] = 4   # fence
ID_TO_TRAINID[17] = 5   # pole
ID_TO_TRAINID[19] = 6   # traffic light
ID_TO_TRAINID[20] = 7   # traffic sign
ID_TO_TRAINID[21] = 8   # vegetation
ID_TO_TRAINID[22] = 9   # terrain
ID_TO_TRAINID[23] = 10  # sky
ID_TO_TRAINID[24] = 11  # person
ID_TO_TRAINID[25] = 12  # rider
ID_TO_TRAINID[26] = 13  # car
ID_TO_TRAINID[27] = 14  # truck
ID_TO_TRAINID[28] = 15  # bus
ID_TO_TRAINID[31] = 16  # train
ID_TO_TRAINID[32] = 17  # motorcycle
ID_TO_TRAINID[33] = 18  # bicycle

class HFCityscapesDataset(Dataset):
    def __init__(self, split='train', image_size=256, repo_id='Chris1/cityscapes'):
        self.ds = load_dataset(repo_id)[split]
        self.image_size = image_size
        self.resize_img = transforms.Resize((image_size, image_size), interpolation=Image.BICUBIC)
        self.resize_mask = transforms.Resize((image_size, image_size), interpolation=Image.NEAREST)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        ex = self.ds[idx]
        img = ex['image']
        msk = ex['semantic_segmentation']
        img = self.resize_img(img.convert('RGB'))
        if not isinstance(msk, Image.Image):
            msk = Image.fromarray(np.array(msk))
        msk = self.resize_mask(msk)
        arr = np.array(msk, dtype=np.int64)
        if arr.ndim == 3:
            arr = arr[:, :, 0]
        arr = ID_TO_TRAINID[arr]
        msk = torch.from_numpy(arr).long()
        return img, msk

@torch.no_grad()
def evaluate_cityscapes(model, dataloader, criterion, num_classes: int, ignore_index: int, device='cuda'):
    model.eval()
    val_loss = 0.0
    k = num_classes
    conf = torch.zeros((k, k), dtype=torch.int64, device=device)

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
        valid = (t != ignore_index) & (t >= 0) & (t < k)
        if valid.any():
            t = t[valid]
            p = p[valid].clamp(min=0, max=k-1)
            n = k
            idx = (t * n + p).view(-1)
            binc = torch.bincount(idx, minlength=n * n)
            conf += binc.view(n, n)

    conf = conf.cpu().numpy()
    inter = np.diag(conf)
    union = conf.sum(1) + conf.sum(0) - inter
    valid = union > 0
    miou = float((inter[valid] / (union[valid] + 1e-10)).mean()) if valid.any() else 0.0
    val_loss = val_loss / max(1, len(dataloader))
    return val_loss, miou

def parse_args():
    p = argparse.ArgumentParser("Cityscapes segmentation using Falcon Vision backbone")
    p.add_argument("--ckpt_path", type=str, required=True, help="Path to checkpoint")
    p.add_argument("--feature_type", type=str, default="dinov3", choices=["dinov3", "amoe", "siglip2"])
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--num_classes", type=int, default=19)
    p.add_argument("--ignore_index", type=int, default=255)
    p.add_argument("--out_dir", type=str, default=None)
    p.add_argument("--hf_repo", type=str, default="Chris1/cityscapes")
    p.add_argument("--image_size", type=int, default=256)
    return p.parse_args()

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset = HFCityscapesDataset(split='train', image_size=args.image_size, repo_id=args.hf_repo)
    val_dataset = HFCityscapesDataset(split='validation', image_size=args.image_size, repo_id=args.hf_repo)
    test_dataset = HFCityscapesDataset(split='test', image_size=args.image_size, repo_id=args.hf_repo)

    # Calculate max patches based on image size (patch size is 16)
    max_patches = (args.image_size // 16) ** 2

    backbone, image_processor = build_backbone_and_processor(
        ckpt_path=args.ckpt_path,
        device=device,
        feature_type=args.feature_type,
    )

    feature_dim = FEATURE_DIM_DICT[args.feature_type]

    seg_model = AMOELinearSeg(
        backbone, num_classes=args.num_classes,
        feature_dim=feature_dim, feature_type=args.feature_type,
        image_size=args.image_size
    ).to(device).to(dtype=torch.bfloat16)

    collate = make_collate_fn(image_processor, max_num_patches=max_patches)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, collate_fn=collate
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, collate_fn=collate
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, collate_fn=collate
    )

    criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_index)
    optimizer = optim.AdamW(seg_model.head.parameters(), lr=args.lr, weight_decay=1e-3)

    best_miou = None
    best_epoch = -1
    best_head_state = None
    
    for epoch in range(args.epochs):
        seg_model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            spatial_shape = batch["spatial_shape"].to(device, non_blocking=True)
            targets = batch["targets"].to(device, non_blocking=True)
            logits = seg_model(pixel_values, spatial_shape)

            optimizer.zero_grad(set_to_none=True)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= max(1, len(train_loader))
        val_loss, val_miou = evaluate_cityscapes(seg_model, val_loader, criterion, num_classes=args.num_classes, ignore_index=args.ignore_index, device=device)
        if (best_miou is None) or (val_miou > best_miou):
            best_miou = val_miou
            best_epoch = epoch + 1
            best_head_state = {k: v.cpu().clone() for k, v in seg_model.head.state_dict().items()}

        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | mIoU(val): {val_miou:.4f} | Best: {best_miou:.4f} @ epoch {best_epoch}")

    if best_head_state is not None:
        seg_model.head.load_state_dict(best_head_state)

    out_dir = args.out_dir or "evals"
    os.makedirs(out_dir, exist_ok=True)

    result = {
        "dataset": "cityscapes",
        "task": "segmentation",
        "feature_type": args.feature_type,
        "metric_name": "mIoU",
        "best_val_mIoU": float(best_miou) if best_miou is not None else None,
        "best_epoch": best_epoch,
        "ckpt_path": args.ckpt_path,
        "num_classes": args.num_classes,
        "ignore_index": args.ignore_index,
    }
    json_path = os.path.join(out_dir, f"cityscapes_segmentation_{args.feature_type}.json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote results to {json_path}")
    print(f"BEST_VAL_MIOU: {best_miou:.4f} @ epoch {best_epoch}")

if __name__ == "__main__":
    main()
