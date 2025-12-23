from tqdm.auto import tqdm
from PIL import Image
import json
import os
import argparse

import torch
torch.set_num_threads(1)

from datasets import load_dataset

from utils import (
    build_model_and_io,
    extract_embeddings_multi,
    combine_logits,
    compute_retrieval_metrics_from_similarity,
)


def main():
    parser = argparse.ArgumentParser(description='MSCOCO retrieval with multi-teacher DistilTransformer (DINOv3 + SigLIP2)')
    parser.add_argument('--ckpt_path', type=str, required=True, help='Path to DistilTransformer checkpoint')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save results JSON')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for processing')
    parser.add_argument('--device', type=str, default="cuda:0", help='Device to use for inference')
    parser.add_argument('--max_pixels_sqrt', type=int, default=256, help='sqrt(max pixels) for patching')
    
    # DINOv3/dinotxt args
    parser.add_argument('--dinov3_repo_dir', type=str, required=True, help='Local DINOv3 repo dir for torch.hub.load')
    parser.add_argument('--dinotxt_weights', type=str, required=True, help='Path to dinotxt weights .pt')
    parser.add_argument('--dinov3_backbone_weights', type=str, required=True, help='Path to DINOv3 backbone weights .pth')
    parser.add_argument('--dinov3_model_name', type=str, default='dinov3_vitl16', help='Real DINOv3 model name for torch.hub')
    parser.add_argument('--siglip2_model_name', type=str, default='google/siglip2-so400m-patch16-naflex', help='SigLIP2 model name for text encoder')
    
    # Dataset
    parser.add_argument('--dataset_path', type=str, default="nlphuji/mscoco_2014_5k_test_image_text_retrieval", help='HF Dataset path or local path')
    parser.add_argument('--split', type=str, default="test", help='Split to evaluate on (validation/test)')

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Build model, tokenizer, image_processor
    model, image_processor = build_model_and_io(
        ckpt_path=args.ckpt_path,
        device=args.device,
        max_pixels_sqrt=args.max_pixels_sqrt
    )
    device = model.device

    # Load dinotxt via torch.hub
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

    print(f"Loading {args.dataset_path} dataset (split={args.split})...")
    # For dandelin/coco_2017, validation split is usually used for evaluation if test is not available or unlabelled.
    # But usually 5k test split is what we want (karpathy test).
    # dandelin/coco_2017 has train, validation, test.
    dataset = load_dataset(args.dataset_path, split=args.split)
    print(f"Dataset size: {len(dataset)} images")

    images = []
    all_texts = []
    image_to_texts_map = {}
    text_idx = 0
    
    # Process dataset into lists
    # Note: COCO HF dataset usually has 'image' (PIL) and 'captions' (list of strings) or 'caption'
    for img_idx, item in enumerate(tqdm(dataset, desc="Preparing data")):
        img = item['image']
        if not isinstance(img, Image.Image):
            if isinstance(img, str):
                img = Image.open(img)
        images.append(img)

        # Handle 'caption' or 'captions'
        if 'captions' in item:
            captions = item['captions']
        elif 'caption' in item:
            captions = item['caption']
        else:
            # Fallback
            captions = []
            
        if isinstance(captions, str):
            captions = [captions]
            
        text_indices = []
        for caption in captions:
            all_texts.append(caption)
            text_indices.append(text_idx)
            text_idx += 1
        image_to_texts_map[img_idx] = text_indices

    print(f"Total images: {len(images)}")
    print(f"Total texts: {len(all_texts)}")

    image_embeddings_dino, text_embeddings_dino, image_embeddings_siglip, text_embeddings_siglip = extract_embeddings_multi(
        model=model,
        dinotxt=dinotxt,
        dinotxt_tokenizer=dinotxt_tokenizer,
        image_processor=image_processor,
        images=images,
        texts=all_texts,
        device=args.device,
        bs=args.batch_size,
        siglip2_model_name=args.siglip2_model_name,
        max_pixels_sqrt=args.max_pixels_sqrt,
    )

    print(f"Image embeddings (DINO) shape: {image_embeddings_dino.shape}")
    print(f"Text embeddings (DINO) shape: {text_embeddings_dino.shape}")
    print(f"Image embeddings (SigLIP2) shape: {image_embeddings_siglip.shape}")
    print(f"Text embeddings (SigLIP2) shape: {text_embeddings_siglip.shape}")

    # Precompute similarity (logits) matrices for ensembling
    print("Precomputing similarity (logits) matrices for ensembling...")
    # Use chunked computation to avoid bottleneck
    from utils import compute_similarity_matrix_chunked
    sim_dino = compute_similarity_matrix_chunked(image_embeddings_dino, text_embeddings_dino, device=args.device)
    sim_siglip = compute_similarity_matrix_chunked(image_embeddings_siglip, text_embeddings_siglip, device=args.device)
    print("Similarity (logits) matrices precomputed")

    # Ensemble logits (entropy-weighted, T=1 like imagenet.py)
    logits_ens = combine_logits(
        sim_dino, sim_siglip,
        T_dino=1.0, T_siglip=1.0,
        mode="entropy_weighted",
        alpha=0.5, beta=5.0
    )

    # Compute retrieval metrics for both heads and ensemble
    print("\n" + "="*60)
    print("EVALUATING DINOv3/dinotxt HEAD")
    print("="*60)
    t2i_dino, i2t_dino = compute_retrieval_metrics_from_similarity(
        sim_dino, image_to_texts_map
    )

    print("\n" + "="*60)
    print("EVALUATING SigLIP2")
    print("="*60)
    t2i_siglip, i2t_siglip = compute_retrieval_metrics_from_similarity(
        sim_siglip, image_to_texts_map
    )

    print("\n" + "="*60)
    print("EVALUATING ENSEMBLE (dinotxt + SigLIP2, entropy-weighted)")
    print("="*60)
    t2i_ens, i2t_ens = compute_retrieval_metrics_from_similarity(
        logits_ens, image_to_texts_map
    )

    print("\n" + "="*60)
    print("MSCOCO EVALUATION RESULTS")
    print("="*60)
    print(f"Model: DistilTransformerMultiTeacher ({args.ckpt_path})")
    print(f"Test images: {len(images)}")
    print(f"Test captions: {len(all_texts)}")
    print("\n" + "-"*40)
    print("DINOv3/dinotxt:")
    print("-"*40)
    print("Text-to-Image Retrieval (T→I):")
    print(f"  Recall@1:  {t2i_dino[1]:.1f}")
    print(f"  Recall@5:  {t2i_dino[5]:.1f}")
    print(f"  Recall@10: {t2i_dino[10]:.1f}")
    print("Image-to-Text Retrieval (I→T):")
    print(f"  Recall@1:  {i2t_dino[1]:.1f}")
    print(f"  Recall@5:  {i2t_dino[5]:.1f}")
    print(f"  Recall@10: {i2t_dino[10]:.1f}")

    print("\n" + "-"*40)
    print("SigLIP2:")
    print("-"*40)
    print("Text-to-Image Retrieval (T→I):")
    print(f"  Recall@1:  {t2i_siglip[1]:.1f}")
    print(f"  Recall@5:  {t2i_siglip[5]:.1f}")
    print(f"  Recall@10: {t2i_siglip[10]:.1f}")
    print("Image-to-Text Retrieval (I→T):")
    print(f"  Recall@1:  {i2t_siglip[1]:.1f}")
    print(f"  Recall@5:  {i2t_siglip[5]:.1f}")
    print(f"  Recall@10: {i2t_siglip[10]:.1f}")

    print("\n" + "-"*40)
    print("ENSEMBLE (entropy-weighted):")
    print("-"*40)
    print("Text-to-Image Retrieval (T→I):")
    print(f"  Recall@1:  {t2i_ens[1]:.1f}")
    print(f"  Recall@5:  {t2i_ens[5]:.1f}")
    print(f"  Recall@10: {t2i_ens[10]:.1f}")
    print("Image-to-Text Retrieval (I→T):")
    print(f"  Recall@1:  {i2t_ens[1]:.1f}")
    print(f"  Recall@5:  {i2t_ens[5]:.1f}")
    print(f"  Recall@10: {i2t_ens[10]:.1f}")

    results = {
        "checkpoint_path": args.ckpt_path,
        "num_images": len(images),
        "num_captions": len(all_texts),
        "dinotxt": {
            "standard": {
                "text_to_image_retrieval": {"recall_at_1": t2i_dino[1], "recall_at_5": t2i_dino[5], "recall_at_10": t2i_dino[10]},
                "image_to_text_retrieval": {"recall_at_1": i2t_dino[1], "recall_at_5": i2t_dino[5], "recall_at_10": i2t_dino[10]},
            },
        },
        "siglip2": {
            "standard": {
                "text_to_image_retrieval": {"recall_at_1": t2i_siglip[1], "recall_at_5": t2i_siglip[5], "recall_at_10": t2i_siglip[10]},
                "image_to_text_retrieval": {"recall_at_1": i2t_siglip[1], "recall_at_5": i2t_siglip[5], "recall_at_10": i2t_siglip[10]},
            },
        },
        "ensemble_entropy_weighted": {
            "standard": {
                "text_to_image_retrieval": {"recall_at_1": t2i_ens[1], "recall_at_5": t2i_ens[5], "recall_at_10": t2i_ens[10]},
                "image_to_text_retrieval": {"recall_at_1": i2t_ens[1], "recall_at_5": i2t_ens[5], "recall_at_10": i2t_ens[10]},
            },
        },
    }

    json_filename = f"mscoco.json"
    json_path = os.path.join(args.output_dir, json_filename)
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {json_path}")


if __name__ == "__main__":
    main()
