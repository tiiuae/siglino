import argparse
import os
import sys
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm.auto import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

from open_clip import OPENAI_IMAGENET_TEMPLATES as OPENAI_TEMPLATES

# Import shared evaluation utilities
from utils import (
    make_collate_fn,
    build_model_and_io,
    process_preprocessed_batch,
    adapt_patches_with_dinov3_head,
    combine_logits,
    compute_text_embeddings_dinotxt,
    compute_text_embeddings_siglip2,
    average_embeddings_over_templates,
)



class ImageNetDataset(Dataset):
    """Simple ImageNet dataset for evaluation"""

    def __init__(self, images_path, imagenet_mappings_path):
        self.image_paths = sorted(Path(images_path).glob("**/*.JPEG"))

        # Load class mappings
        with open(imagenet_mappings_path) as f:
            imagenet_mappings = json.load(f)
        prompt_mapping = imagenet_mappings["prompt_ready_names"]
        sorted_items = sorted(prompt_mapping.items())
        self.cat_to_id = {cat: idx for idx, (cat, _) in enumerate(sorted_items)}

        print(f"Loaded {len(self.image_paths)} images")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]

        try:
            # Load and convert image
            image = Image.open(image_path)
            if hasattr(image, "mode") and image.mode != "RGB":
                image = image.convert("RGB")

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


def setup_text_embeddings_for_imagenet(
    imagenet_mappings_path,
    dinotxt_model,
    dinotxt_tokenizer,
    siglip2_model_name,
    device,
    use_openai_templates=True,
    text_batch_size=4096,
):
    """Setup text embeddings for ImageNet classification accuracy."""
    with open(imagenet_mappings_path) as f:
        imagenet_mappings = json.load(f)

    prompt_mapping = imagenet_mappings["prompt_ready_names"]
    sorted_items = sorted(prompt_mapping.items())
    class_names = [k for k, _ in sorted_items]
    class_display_names = [prompt_mapping[k].lower() for k in class_names]
    id_to_cat = {idx: cat for idx, cat in enumerate(class_names)}

    if use_openai_templates:
        texts_per_class = []
        for name in class_display_names:
            texts = []
            for tmpl in OPENAI_TEMPLATES:
                texts.append(tmpl(name) if callable(tmpl) else tmpl.format(name))
            texts_per_class.append(texts)
        all_texts = [t for texts in texts_per_class for t in texts]
        
        # Dinotxt
        emb_dino = compute_text_embeddings_dinotxt(
            all_texts, dinotxt_model, dinotxt_tokenizer, device, text_batch_size
        )
        emb_dino = average_embeddings_over_templates(emb_dino, len(class_names), len(OPENAI_TEMPLATES))
        
        # SigLIP2
        emb_siglip = compute_text_embeddings_siglip2(
            all_texts, siglip2_model_name, device, text_batch_size
        )
        emb_siglip = average_embeddings_over_templates(emb_siglip, len(class_names), len(OPENAI_TEMPLATES))
    else:
        prompt_list = [f"an image of {name}" for name in class_display_names]
        emb_dino = compute_text_embeddings_dinotxt(
            prompt_list, dinotxt_model, dinotxt_tokenizer, device, text_batch_size
        )
        emb_siglip = compute_text_embeddings_siglip2(
            prompt_list, siglip2_model_name, device, text_batch_size
        )

    return emb_dino, emb_siglip, id_to_cat


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser("ImageNet Classification Accuracy using DINOv3/SigLIP2 multi-teacher distilled model")
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    # Data paths
    parser.add_argument("--images_path", type=str, required=True, help="Path to the ImageNet images")
    parser.add_argument("--imagenet_mappings", type=str, required=True, help="Path to the ImageNet mappings JSON file")

    # Processing parameters
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--prefetch_factor", type=int, default=1)
    parser.add_argument("--pin_memory", type=lambda x: str(x).lower() == "true", default=True)
    parser.add_argument("--persistent_workers", type=lambda x: str(x).lower() == "true", default=False)

    # DINOv3/dinotxt args
    parser.add_argument('--dinov3_repo_dir', type=str, required=True, help='Local DINOv3 repo dir for torch.hub.load')
    parser.add_argument('--dinotxt_weights', type=str, required=True, help='Path to dinotxt weights .pt')
    parser.add_argument('--dinov3_backbone_weights', type=str, required=True, help='Path to DINOv3 backbone weights .pth')
    
    # SigLIP2 args
    parser.add_argument("--siglip2_model_name", type=str, default="google/siglip2-so400m-patch16-naflex")

    # Text embedding options (OpenAI template averaging)
    parser.add_argument("--text_batch_size", type=int, default=4096, help="Batch size for encoding text prompts (templates x classes).")
    parser.add_argument("--no_openai_templates", action="store_true", help="Disable OpenAI template averaging; use a single simple prompt per class.")

    # Device and model IO
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--max_pixels_sqrt', type=int, default=768)

    args = parser.parse_args()
    args.max_pixels = args.max_pixels_sqrt ** 2
    os.makedirs(args.output_dir, exist_ok=True)

    # Distributed setup (auto-detect torchrun)
    using_distributed = False
    rank = 0
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        dist.init_process_group(backend='nccl')
        using_distributed = True
        rank = dist.get_rank()
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        torch.cuda.set_device(local_rank)
        args.device = f"cuda:{local_rank}"

    # Load model/tokenizer and image processor
    model, image_processor = build_model_and_io(
        ckpt_path=args.ckpt_path,
        device=args.device,
        max_pixels=args.max_pixels,
    )
    device = model.device

    # Load dinotxt (DINOv3) via torch.hub
    if rank == 0:
        print("Loading dinotxt (DINOv3) via torch.hub ...")
    dinotxt, dinotxt_tokenizer = torch.hub.load(
        args.dinov3_repo_dir,
        'dinov3_vitl16_dinotxt_tet1280d20h24l',
        source='local',
        dinotxt_weights=args.dinotxt_weights,
        backbone_weights=args.dinov3_backbone_weights,
    )
    dinotxt = dinotxt.to(args.device, dtype=torch.bfloat16)
    dinotxt.eval()

    # Prepare text embeddings for ImageNet classes
    text_embeds_dinotxt, text_embeds_siglip2, id_to_cat = setup_text_embeddings_for_imagenet(
        args.imagenet_mappings,
        dinotxt,
        dinotxt_tokenizer,
        args.siglip2_model_name,
        args.device,
        use_openai_templates=not args.no_openai_templates,
        text_batch_size=args.text_batch_size,
    )

    dataset = ImageNetDataset(args.images_path, args.imagenet_mappings)
    sampler = DistributedSampler(dataset, shuffle=False) if using_distributed else None
    
    # NOTE: make_collate_fn signature updated in utils.py (square_crop removed)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        collate_fn=make_collate_fn(image_processor, max_pixels=args.max_pixels),
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor,
    )

    num_samples = 0
    num_correct_dinotxt = 0
    num_correct_siglip2 = 0
    num_correct_ensemble = 0

    if rank == 0:
        print("Starting evaluation...")
    for batch in tqdm(dataloader, desc="Processing batches", disable=(rank != 0)):
        if batch is None:
            continue

        results = process_preprocessed_batch(batch, model, device)

        # === DINOv3/dinotxt scoring ===
        image_embeds_dino = adapt_patches_with_dinov3_head(
            results['summaries']['dinov3'],
            results['patch_tokens_list_dinov3'],
            dinotxt,
        )
        image_embeds_dino = image_embeds_dino / image_embeds_dino.norm(p=2, dim=-1, keepdim=True)
        image_embeds_dino = image_embeds_dino.to(dtype=torch.bfloat16)
        text_embeds_dino = text_embeds_dinotxt[:, :image_embeds_dino.shape[-1]]
        logits_per_text_dino = image_embeds_dino @ text_embeds_dino.T
        pred_ids_dino = logits_per_text_dino.argmax(dim=1)

        # === SigLIP2 scoring ===
        image_embeds_siglip = results['summaries']['siglip2']
        image_embeds_siglip = image_embeds_siglip / image_embeds_siglip.norm(p=2, dim=-1, keepdim=True)
        image_embeds_siglip = image_embeds_siglip.to(dtype=torch.bfloat16)
        logits_per_text_siglip = image_embeds_siglip @ text_embeds_siglip2.T
        pred_ids_siglip = logits_per_text_siglip.argmax(dim=1)

        # === Ensemble (temperature-calibrated + entropy-weighted) ===
        T_dino = 1.0
        T_siglip = 1.0
        logits_ens = combine_logits(
            logits_per_text_dino, logits_per_text_siglip,
            T_dino=T_dino, T_siglip=T_siglip,
            mode="entropy_weighted",
            alpha=0.5, beta=5.0
        )
        pred_ids_ens = logits_ens.argmax(dim=1)

        for pred_id_dino, pred_id_siglip, pred_id_ens, gt_cat in zip(
            pred_ids_dino, pred_ids_siglip, pred_ids_ens, results['categories']
        ):
            cat_dino = id_to_cat[pred_id_dino.item()]
            cat_siglip = id_to_cat[pred_id_siglip.item()]
            cat_ens = id_to_cat[pred_id_ens.item()]
            if cat_dino == gt_cat:
                num_correct_dinotxt += 1
            if cat_siglip == gt_cat:
                num_correct_siglip2 += 1
            if cat_ens == gt_cat:
                num_correct_ensemble += 1
            num_samples += 1

    # Aggregate across processes
    if using_distributed:
        totals = torch.tensor(
            [num_correct_dinotxt, num_correct_siglip2, num_correct_ensemble, num_samples],
            device=args.device, dtype=torch.long
        )
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        num_correct_dinotxt, num_correct_siglip2, num_correct_ensemble, num_samples = totals.tolist()

    # === SAVE/PRINT RESULTS (rank 0 only) ===
    if (not using_distributed) or (rank == 0):
        accuracy_dinotxt = num_correct_dinotxt / num_samples if num_samples > 0 else 0.0
        accuracy_siglip2 = num_correct_siglip2 / num_samples if num_samples > 0 else 0.0
        accuracy_ensemble = num_correct_ensemble / num_samples if num_samples > 0 else 0.0
        print(f"DINOv3/dinotxt ImageNet Accuracy: {accuracy_dinotxt:.4f} ({num_correct_dinotxt}/{num_samples})")
        print(f"SigLIP2 ImageNet Accuracy: {accuracy_siglip2:.4f} ({num_correct_siglip2}/{num_samples})")
        print(f"Ensemble (entropy-weighted) ImageNet Accuracy: {accuracy_ensemble:.4f} ({num_correct_ensemble}/{num_samples})")

        accuracy_results = {
            "checkpoint_path": args.ckpt_path,
            "num_correct_dinotxt": num_correct_dinotxt,
            "num_correct_siglip2": num_correct_siglip2,
            "num_correct_ensemble": num_correct_ensemble,
            "num_samples": num_samples,
            "accuracy_dinotxt": accuracy_dinotxt,
            "accuracy_siglip2": accuracy_siglip2,
            "accuracy_ensemble": accuracy_ensemble,
            "accuracy_dinotxt_percent": accuracy_dinotxt * 100.0,
            "accuracy_siglip2_percent": accuracy_siglip2 * 100.0,
            "accuracy_ensemble_percent": accuracy_ensemble * 100.0,
        }

        with open(os.path.join(args.output_dir, f"imagenet.json"), 'w') as f:
            json.dump(accuracy_results, f, indent=2)
        print(f"Results saved to: {args.output_dir}")

    if using_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()