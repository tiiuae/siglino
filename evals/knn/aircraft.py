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


def _read_list(list_file: str):
    ids = []
    with open(list_file, "r") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                ids.append(ln.split(" ")[0])
    return ids


def _read_id_to_family(annotations_file: str):
    # Lines: "<id> <family with spaces>"
    id_to_family = {}
    with open(annotations_file, "r") as f:
        for ln in f:
            ln = ln.rstrip("\n")
            if not ln:
                continue
            parts = ln.split(" ", 1)
            if len(parts) < 2:
                continue
            img_id, family = parts[0], parts[1].strip()
            id_to_family[img_id] = family
    return id_to_family

class FGVCAircraftDataset(Dataset):
    """FGVC-Aircraft dataset using family labels."""

    def __init__(self, images_dir: str, list_file: str, annotations_file: str):
        self.images_dir = Path(images_dir)
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images dir not found: {self.images_dir}")

        ids = set(_read_list(list_file))
        id_to_family = _read_id_to_family(annotations_file)
        present_families = sorted({id_to_family[i] for i in ids if i in id_to_family})
        self.cat_to_id = {fam: i for i, fam in enumerate(present_families)}
        self.id_to_cat = {i: fam for fam, i in self.cat_to_id.items()}

        # Collect samples (resolve file extension)
        exts = [".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPG", ".JPEG", ".PNG"]
        samples = []
        for img_id in sorted(ids):
            fam = id_to_family.get(img_id, None)
            if fam is None:
                continue
            
            img_path = None
            for ext in exts:
                cand = self.images_dir / f"{img_id}{ext}"
                if cand.exists():
                    img_path = cand
                    break
            if img_path is None:
                continue
            samples.append((str(img_path), fam, self.cat_to_id[fam]))

        self.samples = samples
        print(f"Loaded {len(self.samples)} images across {len(self.cat_to_id)} families")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, category, class_id = self.samples[idx]
        try:
            image = Image.open(image_path)
            if hasattr(image, "mode") and image.mode != "RGB":
                image = image.convert("RGB")
            return {
                "image": np.array(image),
                "class_id": class_id,
                "category": category,  # family string
                "image_path": image_path,
            }
        except Exception as e:
            print(f"Failed to load {image_path}: {e}")
            return None

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser("FGVC-Aircraft kNN Classification")
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    
    # Data paths
    parser.add_argument("--images_dir", type=str, required=True, help="Directory with images")
    parser.add_argument("--train_list", type=str, required=True, help="Train images list (txt)")
    parser.add_argument("--train_annotations", type=str, required=True, help="Train annotations (txt)")
    parser.add_argument("--val_list", type=str, required=True, help="Val/Test images list (txt)")
    parser.add_argument("--val_annotations", type=str, required=True, help="Val/Test annotations (txt)")
    
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
    train_dataset = FGVCAircraftDataset(args.images_dir, args.train_list, args.train_annotations)
    val_dataset = FGVCAircraftDataset(args.images_dir, args.val_list, args.val_annotations)
    
    # Consistency check: we need same class ID mapping
    # Assuming both splits cover all families or we are lucky with sorting.
    # To be safe, we can enforce val_dataset to use train_dataset's mapping if needed, 
    # but here I'll just check if they differ.
    if train_dataset.cat_to_id != val_dataset.cat_to_id:
        if rank == 0:
            print("[Warning] Train and Val datasets have different class mappings! kNN will fail/be wrong.")
            # Force val to use train mapping
            val_dataset.cat_to_id = train_dataset.cat_to_id
            # Recompute samples with new mapping
            new_samples = []
            for p, fam, _ in val_dataset.samples:
                if fam in train_dataset.cat_to_id:
                    new_samples.append((p, fam, train_dataset.cat_to_id[fam]))
            val_dataset.samples = new_samples
            print(f"[Info] Remapped Val dataset to match Train class IDs. New val size: {len(val_dataset)}")

    num_classes = len(train_dataset.cat_to_id)

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
        with open(os.path.join(args.output_dir, f"aircraft_knn.json"), "w") as f:
            json.dump(results, f, indent=2)

    if using_distributed:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
