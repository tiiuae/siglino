import argparse
import os
import json
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm.auto import tqdm
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from utils import (
    build_model_and_io,
    process_preprocessed_batch,
    build_embeddings,
    _distributed_topk,
    class_votes,
    fuse_votes_entropy_weighted,
    make_collate_fn,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def _parse_cub_classes(classes_file):
    # classes.txt lines: "1 001.Black_footed_Albatross"
    id_to_key = {}
    key_to_id0 = {}
    with open(classes_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            idx_str, key = line.split()
            idx = int(idx_str)  # 1-based
            id_to_key[idx - 1] = key  # 0-based
            key_to_id0[key] = idx - 1
    return id_to_key, key_to_id0

def _parse_cub_images(images_file):
    # images.txt lines: "1 001.Black_footed_Albatross/xxx.jpg"
    imgid_to_relpath = {}
    with open(images_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            idx_str, relpath = line.split()
            imgid_to_relpath[int(idx_str)] = relpath
    return imgid_to_relpath

def _parse_cub_split(split_file):
    # train_test_split.txt lines: "1 0" (0=test, 1=train)
    imgid_to_split = {}
    with open(split_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            idx_str, split_str = line.split()
            imgid_to_split[int(idx_str)] = int(split_str)
    return imgid_to_split

class CUBDataset(Dataset):
    """CUB-200-2011 dataset."""

    def __init__(self, cub_root, split_value=0):
        self.cub_root = Path(cub_root)
        self.images_root = self.cub_root / "images"
        classes_file = self.cub_root / "classes.txt"
        images_file = self.cub_root / "images.txt"
        split_file = self.cub_root / "train_test_split.txt"

        id_to_key, key_to_id0 = _parse_cub_classes(classes_file)
        self.key_to_id0 = key_to_id0
        self.cat_to_id = self.key_to_id0 # Key string to ID

        imgid_to_rel = _parse_cub_images(images_file)
        imgid_to_split = _parse_cub_split(split_file)

        self.samples = []
        for img_id, rel_path in imgid_to_rel.items():
            split = imgid_to_split.get(img_id, None)
            if split is None:
                continue
            if split != split_value:
                continue
            full_path = self.images_root / rel_path
            # category is the class folder name e.g. "001.Black_footed_Albatross"
            category = rel_path.split("/")[0]
            class_id0 = self.key_to_id0[category]
            self.samples.append(
                {
                    "image_path": str(full_path),
                    "category": category,
                    "class_id": class_id0,
                }
            )

        print(f"Loaded {len(self.samples)} CUB images (split={split_value})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        info = self.samples[idx]
        image_path = info["image_path"]
        try:
            image = Image.open(image_path)
            if hasattr(image, "mode") and image.mode != "RGB":
                image = image.convert("RGB")
            return {
                "image": np.array(image),
                "class_id": info["class_id"],
                "category": info["category"],
                "image_path": image_path,
            }
        except Exception as e:
            print(f"Failed to load {image_path}: {e}")
            return None

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser("CUB-200-2011 kNN Classification")
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    
    # Data paths
    parser.add_argument("--cub_root", type=str, required=True, help="CUB-200-2011 root directory")
    
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--prefetch_factor", type=int, default=1)
    parser.add_argument("--pin_memory", type=lambda x: str(x).lower() == "true", default=True)
    parser.add_argument("--persistent_workers", type=lambda x: str(x).lower() == "true", default=False)
    
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max_pixels_sqrt", type=int, default=768)
    
    # kNN params
    parser.add_argument("-k", "--k_neighbors", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--ensemble_beta", type=float, default=5.0)

    args = parser.parse_args()
    args.max_pixels = args.max_pixels_sqrt ** 2
    os.makedirs(args.output_dir, exist_ok=True)

    # Distributed
    using_distributed = False
    rank = 0
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        using_distributed = True
        rank = dist.get_rank()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        args.device = f"cuda:{local_rank}"

    # Model
    model, image_processor = build_model_and_io(
        ckpt_path=args.ckpt_path,
        device=args.device,
        max_pixels=args.max_pixels,
    )
    device = model.device

    # Datasets
    # 1=train, 0=test
    train_dataset = CUBDataset(args.cub_root, split_value=1)
    val_dataset = CUBDataset(args.cub_root, split_value=0)
    
    num_classes = len(train_dataset.key_to_id0)

    train_sampler = DistributedSampler(train_dataset, shuffle=False) if using_distributed else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if using_distributed else None
    
    collate_fn = make_collate_fn(image_processor, max_num_patches=(args.max_pixels_sqrt ** 2 // 16 ** 2))
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None), sampler=train_sampler, collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=args.pin_memory, persistent_workers=args.persistent_workers, prefetch_factor=args.prefetch_factor)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=(val_sampler is None), sampler=val_sampler, collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=args.pin_memory, persistent_workers=args.persistent_workers, prefetch_factor=args.prefetch_factor)

    # Build kNN database
    if rank == 0:
        print("Building kNN index...")
    keys_dino, keys_siglip, key_labels = build_embeddings(train_loader, model, device)

    # Evaluation
    if rank == 0:
        print("Evaluating...")
        
    num_samples = 0
    num_correct_dino = 0
    num_correct_siglip = 0
    num_correct_ens = 0

    for batch in tqdm(val_loader, desc="Evaluating", disable=(rank!=0)):
        if batch is None: continue
        
        out = process_preprocessed_batch(batch, model, device)
        q_dino = out['summaries']['dinov3']
        q_siglip = out['summaries']['siglip2']
        q_dino = q_dino / q_dino.norm(p=2, dim=-1, keepdim=True)
        q_siglip = q_siglip / q_siglip.norm(p=2, dim=-1, keepdim=True)

        sim_dino, lab_dino = _distributed_topk(q_dino, keys_dino, key_labels, args.k_neighbors, using_distributed)
        sim_siglip, lab_siglip = _distributed_topk(q_siglip, keys_siglip, key_labels, args.k_neighbors, using_distributed)

        votes_dino = class_votes(sim_dino, lab_dino, num_classes, args.temperature)
        votes_siglip = class_votes(sim_siglip, lab_siglip, num_classes, args.temperature)

        pred_dino = votes_dino.argmax(dim=1)
        pred_siglip = votes_siglip.argmax(dim=1)
        
        votes_ens = fuse_votes_entropy_weighted(votes_dino, votes_siglip, beta=args.ensemble_beta)
        pred_ens = votes_ens.argmax(dim=1)

        gt = out['class_ids'].to(device)
        
        num_correct_dino += int((pred_dino == gt).sum().item())
        num_correct_siglip += int((pred_siglip == gt).sum().item())
        num_correct_ens += int((pred_ens == gt).sum().item())
        num_samples += int(gt.numel())

    if using_distributed:
        totals = torch.tensor([num_correct_dino, num_correct_siglip, num_correct_ens, num_samples], device=device, dtype=torch.long)
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        num_correct_dino, num_correct_siglip, num_correct_ens, num_samples = totals.tolist()

    if (not using_distributed) or (rank == 0):
        acc_dino = num_correct_dino / num_samples if num_samples > 0 else 0.0
        acc_siglip = num_correct_siglip / num_samples if num_samples > 0 else 0.0
        acc_ens = num_correct_ens / num_samples if num_samples > 0 else 0.0
        print(f"DINOv3 kNN Accuracy: {acc_dino:.4f}")
        print(f"SigLIP2 kNN Accuracy: {acc_siglip:.4f}")
        print(f"Ensemble kNN Accuracy: {acc_ens:.4f}")
        
        ckpt_basename = args.ckpt_path.split("/")[-1]
        results = {
            "accuracy_dinov3": acc_dino,
            "accuracy_siglip2": acc_siglip,
            "accuracy_ensemble": acc_ens,
        }
        with open(os.path.join(args.output_dir, f"cub200_knn.json"), "w") as f:
            json.dump(results, f, indent=2)

    if using_distributed:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
