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


class ImageNetDataset(Dataset):
    """Simple ImageNet dataset for evaluation and kNN indexing"""

    def __init__(self, images_path, imagenet_mappings_path):
        self.image_paths = sorted(Path(images_path).glob("**/*.JPEG"))

        # Load class mappings to verify classes or get IDs
        with open(imagenet_mappings_path) as f:
            imagenet_mappings = json.load(f)
        prompt_mapping = imagenet_mappings["prompt_ready_names"]
        sorted_items = sorted(prompt_mapping.items())
        # Map folder name (wnid) to index 0..999
        self.cat_to_id = {cat: idx for idx, (cat, _) in enumerate(sorted_items)}
        self.num_classes = len(self.cat_to_id)

        print(f"Loaded {len(self.image_paths)} images from {images_path}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        try:
            image = Image.open(image_path)
            if hasattr(image, "mode") and image.mode != "RGB":
                image = image.convert("RGB")
            # Parent folder is category (wnid)
            category = image_path.parent.stem
            class_id = self.cat_to_id[category]
            return {
                'image': np.array(image),
                'class_id': class_id,
                'category': category,
                'image_path': str(image_path)
            }
        except Exception as e:
            print(f"Failed to load {image_path}: {e}")
            return None

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser("ImageNet kNN Classification")
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    # Data paths
    parser.add_argument("--images_path", type=str, required=True, help="Path to the ImageNet validation images")
    parser.add_argument("--train_images_path", type=str, required=True, help="Path to the ImageNet training images")
    parser.add_argument("--imagenet_mappings", type=str, required=True, help="Path to the ImageNet mappings JSON file")
    
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--prefetch_factor", type=int, default=1)
    parser.add_argument("--pin_memory", type=lambda x: str(x).lower() == "true", default=True)
    parser.add_argument("--persistent_workers", type=lambda x: str(x).lower() == "true", default=False)
    
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max_pixels_sqrt", type=int, default=256)
    
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
    train_dataset = ImageNetDataset(args.train_images_path, args.imagenet_mappings)
    val_dataset = ImageNetDataset(args.images_path, args.imagenet_mappings)
    num_classes = train_dataset.num_classes

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
        
        results = {
            "accuracy_dinov3": acc_dino,
            "accuracy_siglip2": acc_siglip,
            "accuracy_ensemble": acc_ens,
        }
        with open(os.path.join(args.output_dir, f"imagenet_knn.json"), "w") as f:
            json.dump(results, f, indent=2)

    if using_distributed:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
