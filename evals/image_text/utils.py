import torch
from tqdm.auto import tqdm
from transformers import AutoProcessor, AutoModel
import sys
import os
from amoe.utils import load_amoe_model




def make_collate_fn(image_processor, max_pixels: int):
    """
    Collate function returning:
      - pixel_values, padding_mask, spatial_shapes (for Falcon Vision)
      - dinov3_images (padded batch for real DINOv3)
      - class_ids, categories, image_paths (metadata)
    """
    # Assuming patch size 16 for calculating max_num_patches
    max_num_patches = max_pixels // (16 * 16)

    def _collate(batch):
        batch = [item for item in batch if item is not None]
        if not batch:
            return None

        # Preprocess images for Falcon Vision
        # ImageProcessor takes list of images (PIL or array)
        images_for_model = [item["image"] for item in batch]
        processed = image_processor(images_for_model, max_num_patches=max_num_patches)

        return {
            "pixel_values": processed["pixel_values"],
            "padding_mask": processed["padding_mask"],
            "spatial_shapes": processed["spatial_shape"],
            "class_ids": torch.tensor([item["class_id"] for item in batch], dtype=torch.long),
            "categories": [item["category"] for item in batch],
            "image_paths": [item["image_path"] for item in batch],
        }

    return _collate


def build_model_and_io(
    ckpt_path: str,
    device: str,
    max_pixels: int,
):
    model, image_processor = load_amoe_model(
        checkpoint_path=ckpt_path,
        device=device,
        max_pixels=max_pixels,
    )
    model = model.to(torch.bfloat16)
    
    return model, image_processor


def process_preprocessed_batch(batch, model, device):
    """Forward pass using standalone Falcon Vision model."""
    pixel_values = batch["pixel_values"].to(device, non_blocking=True).to(torch.bfloat16)
    padding_mask = batch["padding_mask"].to(device, non_blocking=True)
    spatial_shapes = batch["spatial_shapes"].to(device, non_blocking=True)

    with torch.inference_mode():
        outputs = model(
            pixel_values=pixel_values,
            padding_mask=padding_mask,
            spatial_shapes=spatial_shapes,
        )

        summaries = outputs["summary_features"]  # {"dinov3": ..., "siglip2": ...}
        
        # Extract patch tokens (excluding padding) for dinotxt head
        student_patch_dinov3 = outputs["patch_features"]["dinov3"]  # (N, L, D) - includes padding
        
        patch_tokens_list_dinov3 = []
        for i in range(len(student_patch_dinov3)):
            # Determine valid length from padding_mask
            L_valid = int(padding_mask[i].sum().item())
            # For standalone model, output has L tokens.
            # padding_mask has L tokens.
            patch_tokens_list_dinov3.append(student_patch_dinov3[i, :L_valid])

    return {
        "summaries": summaries,
        "patch_tokens_list_dinov3": patch_tokens_list_dinov3,
        "class_ids": batch.get("class_ids"),
        "categories": batch.get("categories"),
        "image_paths": batch.get("image_paths"),
        "dinov3_images": batch.get("dinov3_images"),
    }


def adapt_patches_with_dinov3_head(image_summary_dinov3, patch_tokens_list_dinov3, dinotxt, storage_tokens_batch=None):
    """Adapt tokens via dinotxt.visual_model.head using sequence:
    [CLS] + [optional storage] + [patches]."""
    if not hasattr(dinotxt, "visual_model") or not hasattr(dinotxt.visual_model, "head"):
        return None
    head = dinotxt.visual_model.head
    device = image_summary_dinov3.device
    dtype = image_summary_dinov3.dtype

    image_embeds = []
    for i, tokens in enumerate(patch_tokens_list_dinov3):
        cls_tok = image_summary_dinov3[i].unsqueeze(0)  # (1, D_dino)

        storage_seq = None
        if storage_tokens_batch is not None:
            st = storage_tokens_batch[i]
            storage_seq = st.to(device=device, dtype=dtype)

        seq = tokens
        if storage_seq is not None:
            seq = torch.cat([cls_tok, storage_seq, seq], dim=0)
        else:
            seq = torch.cat([cls_tok, seq], dim=0)

        x = seq.unsqueeze(0)  # (1, L_total, D_dino)
        with torch.inference_mode():
            out = head(x)
        out = torch.cat([out[:, 0, :], out[:, 5:, :].mean(dim=1)], dim=-1)
        image_embeds.append(out)
    return torch.cat(image_embeds, dim=0)

# -----------------------------------------------------------------------------
# Retrieval Logic
# -----------------------------------------------------------------------------

def extract_embeddings_multi(model, dinotxt, dinotxt_tokenizer, image_processor, images, texts, device, real_dinov3=None, bs=32, siglip2_model_name="google/siglip2-so400m-patch16-naflex", max_pixels_sqrt=256):
    """Extract image/text embeddings for both dinotxt and SigLIP2."""
    import math
    
    # Text embeddings with dinotxt
    print("Extracting text embeddings (dinotxt)...")
    text_embeddings_dino = []
    for i in tqdm(range(0, len(texts), bs)):
        batch_texts = texts[i:i+bs]
        with torch.inference_mode():
            text_tokens = dinotxt_tokenizer.tokenize(batch_texts).to(device)
            text_embeds = dinotxt.encode_text(text_tokens)
            text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
            text_embeddings_dino.append(text_embeds.to(dtype=torch.bfloat16).detach().cpu())
        del text_tokens, text_embeds
    text_embeddings_dino = torch.cat(text_embeddings_dino, dim=0)

    # Text embeddings with SigLIP2
    print("Extracting text embeddings (SigLIP2)...")
    siglip_text_tower = AutoModel.from_pretrained(siglip2_model_name).text_model.to(device).to(torch.bfloat16)
    siglip_text_tower.eval()
    processor = AutoProcessor.from_pretrained(siglip2_model_name, max_num_patches=256)
    text_embeddings_siglip = []
    for i in tqdm(range(0, len(texts), bs)):
        batch_texts = texts[i:i+bs]
        ids = []
        for t in batch_texts:
            enc = processor.tokenizer.encode(t)
            inp = torch.nn.functional.pad(
                torch.tensor(enc, dtype=torch.long, device=device),
                (0, 64 - len(enc)),
                value=processor.tokenizer.pad_token_id if hasattr(processor.tokenizer, "pad_token_id") else 0
            )[:64]
            ids.append(inp)
        input_ids = torch.stack(ids, dim=0)
        with torch.inference_mode():
            text_emb = siglip_text_tower(input_ids).pooler_output
            text_emb = text_emb / text_emb.norm(p=2, dim=-1, keepdim=True)
            text_embeddings_siglip.append(text_emb.to(dtype=torch.bfloat16).detach().cpu())
        del input_ids, text_emb
    text_embeddings_siglip = torch.cat(text_embeddings_siglip, dim=0)
    del siglip_text_tower, processor
    torch.cuda.empty_cache()

    # Image embeddings
    print("Extracting image embeddings (MOE + dinotxt head, and SigLIP2 summary)...")
    # Calculate max_num_patches assuming patch_size=16
    max_num_patches = (max_pixels_sqrt ** 2) // (16 * 16)
    image_embeddings_dino = []
    image_embeddings_siglip = []
    
    for i in tqdm(range(0, len(images), bs)):
        batch_images = images[i:i+bs]
        batch_images = [img.convert("RGB") if hasattr(img, "mode") and img.mode != "RGB" else img for img in batch_images]

        # Falcon-style preprocessing
        processed = image_processor(batch_images, max_num_patches=max_num_patches)
        out = process_preprocessed_batch(processed, model, device)
        image_summary_dict = out['summaries']
        image_summary_dinov3 = image_summary_dict["dinov3"]
        image_summary_siglip2 = image_summary_dict["siglip2"]
        patch_tokens_list = out['patch_tokens_list_dinov3']

        # Optional: get storage tokens from real DINOv3
        with torch.inference_mode():
            imgs = processed["pixel_values"].to(device, non_blocking=True)
            with torch.autocast(device_type='cuda', dtype=torch.float32):
                ret = dinotxt.forward_features(imgs, masks=processed["padding_mask"])
            storage_tokens_batch = ret.get('x_storage_tokens', None)

        # Build image embeddings via dinov3 head
        embeds_dino = adapt_patches_with_dinov3_head(
            image_summary_dinov3,
            patch_tokens_list,
            dinotxt,
            storage_tokens_batch=storage_tokens_batch
        )
        if embeds_dino is None:
            embeds_dino = image_summary_dinov3
        embeds_dino = embeds_dino / embeds_dino.norm(p=2, dim=-1, keepdim=True)

        # SigLIP2 image embeddings from summary
        embeds_siglip = image_summary_siglip2 / image_summary_siglip2.norm(p=2, dim=-1, keepdim=True)

        image_embeddings_dino.append(embeds_dino.to(dtype=torch.bfloat16).detach().cpu())
        image_embeddings_siglip.append(embeds_siglip.to(dtype=torch.bfloat16).detach().cpu())

        del processed, out, embeds_dino, embeds_siglip, image_summary_dict, image_summary_dinov3, image_summary_siglip2, patch_tokens_list, storage_tokens_batch

    image_embeddings_dino = torch.cat(image_embeddings_dino, dim=0)
    image_embeddings_siglip = torch.cat(image_embeddings_siglip, dim=0)

    # Align dims if dinotxt head changed dims via concat
    if image_embeddings_dino.shape[-1] != text_embeddings_dino.shape[-1]:
        Dd = image_embeddings_dino.shape[-1]
        text_embeddings_dino = text_embeddings_dino[:, :Dd]
    # Align dims for SigLIP2 if needed
    if image_embeddings_siglip.shape[-1] != text_embeddings_siglip.shape[-1]:
        Ds = image_embeddings_siglip.shape[-1]
        text_embeddings_siglip = text_embeddings_siglip[:, :Ds]

    return image_embeddings_dino, text_embeddings_dino, image_embeddings_siglip, text_embeddings_siglip


def combine_logits(
    logits_dino: torch.Tensor,
    logits_siglip: torch.Tensor,
    T_dino: float = 1.0,
    T_siglip: float = 1.0,
    mode: str = "entropy_weighted",
    alpha: float = 0.5,
    beta: float = 5.0,
) -> torch.Tensor:
    """Ensemble logits from DINOv3 and SigLIP2 branches."""
    p1 = torch.softmax(logits_dino / T_dino, dim=-1)
    p2 = torch.softmax(logits_siglip / T_siglip, dim=-1)

    if mode == "avg":
        return 0.5 * (logits_dino / T_dino) + 0.5 * (logits_siglip / T_siglip)

    if mode == "weighted":
        return alpha * (logits_dino / T_dino) + (1.0 - alpha) * (logits_siglip / T_siglip)

    if mode == "entropy_weighted":
        H1 = -(p1.clamp_min(1e-8) * p1.clamp_min(1e-8).log()).sum(dim=-1, keepdim=True)
        H2 = -(p2.clamp_min(1e-8) * p2.clamp_min(1e-8).log()).sum(dim=-1, keepdim=True)
        w1 = torch.softmax(torch.cat([-beta * H1, -beta * H2], dim=-1), dim=-1)[..., 0:1]
        w2 = 1.0 - w1
        return w1 * (logits_dino / T_dino) + w2 * (logits_siglip / T_siglip)

    raise ValueError(f"Unknown combine mode: {mode}")


def compute_text_embeddings_dinotxt(
    texts: list[str],
    dinotxt_model,
    dinotxt_tokenizer,
    device: str,
    batch_size: int = 1024
):
    """Compute normalized text embeddings using dinotxt."""
    embeds_chunks = []
    with torch.inference_mode():
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding texts (dinotxt)"):
            curr = texts[i : i + batch_size]
            tokens = dinotxt_tokenizer.tokenize(curr).to(device)
            out = dinotxt_model.encode_text(tokens)
            out = out / out.norm(p=2, dim=-1, keepdim=True)
            embeds_chunks.append(out.to(torch.bfloat16))
    return torch.cat(embeds_chunks, dim=0)


def compute_text_embeddings_siglip2(
    texts: list[str],
    model_name: str,
    device: str,
    batch_size: int = 1024
):
    """Compute normalized text embeddings using SigLIP2."""
    text_tower = AutoModel.from_pretrained(model_name).text_model.to(device).to(torch.bfloat16)
    text_tower.eval()
    processor = AutoProcessor.from_pretrained(model_name, max_num_patches=256)

    pad_id = processor.tokenizer.pad_token_id if hasattr(processor.tokenizer, "pad_token_id") else 0
    max_len = 64
    
    ids_list = []
    for txt in texts:
        enc = processor.tokenizer.encode(txt)
        t = torch.tensor(enc[:max_len], dtype=torch.long)
        if t.numel() < max_len:
            t = torch.nn.functional.pad(t, (0, max_len - t.numel()), value=pad_id)
        ids_list.append(t)
    ids = torch.stack(ids_list, dim=0)

    embeds_chunks = []
    with torch.inference_mode():
        for i in tqdm(range(0, ids.size(0), batch_size), desc="Encoding texts (SigLIP2)"):
            curr = ids[i : i + batch_size].to(device)
            out = text_tower(curr).pooler_output
            out = out / out.norm(p=2, dim=-1, keepdim=True)
            embeds_chunks.append(out.to(torch.bfloat16))
            
    del text_tower, processor
    torch.cuda.empty_cache()
    return torch.cat(embeds_chunks, dim=0)


def average_embeddings_over_templates(embeddings, num_classes, num_templates):
    """Reshape [C*T, D] -> [C, T, D] -> mean -> [C, D], then normalize."""
    orig_dtype = embeddings.dtype
    embeddings = embeddings.reshape(num_classes, num_templates, -1)
    embeddings = embeddings.mean(dim=1)
    embeddings = embeddings / embeddings.norm(p=2, dim=-1, keepdim=True)
    return embeddings.to(orig_dtype)
