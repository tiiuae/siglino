import torch
import math
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm.auto import tqdm
import torchvision.transforms as transforms
from transformers import AutoProcessor, AutoModel
import sys
import os
from torch.utils.data import DataLoader, Dataset
from amoe import load_amoe_model


def build_model_and_io(
    ckpt_path: str,
    device: str,
    max_pixels_sqrt: int = 256,
):
    model, image_processor = load_amoe_model(
        checkpoint_path=ckpt_path,
        device=device,
        max_pixels=max_pixels_sqrt**2,
    )
    model = model.to(torch.bfloat16)
    
    return model, image_processor

# -----------------------------------------------------------------------------
# Forward Pass & Adaptation
# -----------------------------------------------------------------------------

def adapt_patches_with_dinov3_head(image_summary, patch_tokens_list, dinotxt):
    if not hasattr(dinotxt, "visual_model") or not hasattr(dinotxt.visual_model, "head"):
        return None
    head = dinotxt.visual_model.head
    device = image_summary.device
    dtype = image_summary.dtype

    image_embeds = []
    for i, tokens in enumerate(patch_tokens_list):
        cls_tok = image_summary[i].unsqueeze(0)  # (1, D)

        seq = tokens
        seq = torch.cat([cls_tok, seq], dim=0)

        x = seq.unsqueeze(0)
        with torch.inference_mode():
            out = head(x)
        out = torch.cat([out[:, 0, :], out[:, 5:, :].mean(dim=1)], dim=-1)
        image_embeds.append(out)
    return torch.cat(image_embeds, dim=0)

@torch.no_grad()
def process_preprocessed_batch(processed, model, device):
    pixel_values = processed["pixel_values"].to(device, non_blocking=True).to(torch.bfloat16)
    padding_mask = processed["padding_mask"].to(device, non_blocking=True)
    spatial_shapes = processed["spatial_shape"].to(device, non_blocking=True)

    outputs = model(
        pixel_values=pixel_values,
        padding_mask=padding_mask,
        spatial_shapes=spatial_shapes,
    )

    summaries = outputs["summary_features"]
    student_patch_dinov3 = outputs["patch_features"]["dinov3"]

    patch_tokens_list_dinov3 = []
    for i in range(len(student_patch_dinov3)):
        L_valid = int(padding_mask[i].sum().item())
        patch_tokens_list_dinov3.append(student_patch_dinov3[i, :L_valid])

    return {
        'summaries': summaries,
        'patch_tokens_list_dinov3': patch_tokens_list_dinov3,
    }

# -----------------------------------------------------------------------------
# Retrieval Logic
# -----------------------------------------------------------------------------

class SimpleListDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

class SiglipTextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len=64, pad_id=0):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.pad_id = pad_id
        
    def __len__(self):
        return len(self.texts)
        
    def __getitem__(self, idx):
        txt = self.texts[idx]
        enc = self.tokenizer.encode(txt)
        t = torch.tensor(enc[:self.max_len], dtype=torch.long)
        if t.numel() < self.max_len:
            t = torch.nn.functional.pad(t, (0, self.max_len - t.numel()), value=self.pad_id)
        return t

class ImageDataset(Dataset):
    def __init__(self, images):
        self.images = images
        
    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, idx):
        img = self.images[idx]
        if hasattr(img, "mode") and img.mode != "RGB":
            img = img.convert("RGB")
        return img

def collate_pil_list(batch):
    return batch

def extract_embeddings_multi(model, dinotxt, dinotxt_tokenizer, image_processor, images, texts, device, bs=512, siglip2_model_name="google/siglip2-so400m-patch16-naflex", max_pixels_sqrt=256, num_workers=4):
    """Extract image/text embeddings for both dinotxt and SigLIP2."""
    import math
    
    # -------------------------------------------------------------------------
    # 1. Text embeddings with dinotxt
    # -------------------------------------------------------------------------
    print("Extracting text embeddings (dinotxt)...")
    text_dataset = SimpleListDataset(texts)
    text_loader = DataLoader(text_dataset, batch_size=bs, shuffle=False, num_workers=num_workers, collate_fn=lambda x: x)
    
    text_embeddings_dino = []
    for batch_texts in tqdm(text_loader, desc="DINOtxt texts"):
        with torch.inference_mode():
            text_tokens = dinotxt_tokenizer.tokenize(batch_texts).to(device)
            text_embeds = dinotxt.encode_text(text_tokens)
            text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
            text_embeddings_dino.append(text_embeds.to(dtype=torch.bfloat16).detach().cpu())
        del text_tokens, text_embeds
    text_embeddings_dino = torch.cat(text_embeddings_dino, dim=0)

    # -------------------------------------------------------------------------
    # 2. Text embeddings with SigLIP2
    # -------------------------------------------------------------------------
    print("Extracting text embeddings (SigLIP2)...")
    siglip_text_tower = AutoModel.from_pretrained(siglip2_model_name).text_model.to(device).to(torch.bfloat16)
    siglip_text_tower.eval()
    processor = AutoProcessor.from_pretrained(siglip2_model_name, max_num_patches=256)
    
    pad_id = processor.tokenizer.pad_token_id if hasattr(processor.tokenizer, "pad_token_id") else 0
    siglip_dataset = SiglipTextDataset(texts, processor.tokenizer, max_len=64, pad_id=pad_id)
    # num_workers=0 here because tokenizer might not pickle well or be thread-safe if from HF tokenizers fast
    # But usually it's fine. Using 0 to be safe unless we need speed. 
    # Actually, tokenization is slow, so workers=num_workers is better if it works.
    # Let's try num_workers.
    siglip_loader = DataLoader(siglip_dataset, batch_size=bs, shuffle=False, num_workers=num_workers)

    text_embeddings_siglip = []
    for input_ids in tqdm(siglip_loader, desc="SigLIP2 texts"):
        input_ids = input_ids.to(device)
        with torch.inference_mode():
            text_emb = siglip_text_tower(input_ids).pooler_output
            text_emb = text_emb / text_emb.norm(p=2, dim=-1, keepdim=True)
            text_embeddings_siglip.append(text_emb.to(dtype=torch.bfloat16).detach().cpu())
        del input_ids, text_emb
    text_embeddings_siglip = torch.cat(text_embeddings_siglip, dim=0)
    del siglip_text_tower, processor
    torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # 3. Image embeddings
    # -------------------------------------------------------------------------
    print("Extracting image embeddings (MOE + dinotxt head, and SigLIP2 summary)...")
    max_num_patches = (max_pixels_sqrt**2) // (16 * 16)
    
    image_dataset = ImageDataset(images)
    # Using collate_pil_list to get list of PIL images, avoiding default collate stack
    image_loader = DataLoader(image_dataset, batch_size=bs, shuffle=False, num_workers=num_workers, collate_fn=collate_pil_list)

    image_embeddings_dino = []
    image_embeddings_siglip = []
    
    for batch_images in tqdm(image_loader, desc="Images"):
        # Falcon-style preprocessing
        processed = image_processor(batch_images, max_num_patches=max_num_patches)
        out = process_preprocessed_batch(processed, model, device)
        image_summary_dict = out['summaries']
        image_summary_dinov3 = image_summary_dict["dinov3"]
        image_summary_siglip2 = image_summary_dict["siglip2"]
        patch_tokens_list = out['patch_tokens_list_dinov3']

        # Build image embeddings via dinov3 head
        embeds_dino = adapt_patches_with_dinov3_head(
            image_summary_dinov3,
            patch_tokens_list,
            dinotxt,
        )
        if embeds_dino is None:
            embeds_dino = image_summary_dinov3
        embeds_dino = embeds_dino / embeds_dino.norm(p=2, dim=-1, keepdim=True)

        # SigLIP2 image embeddings from summary
        embeds_siglip = image_summary_siglip2 / image_summary_siglip2.norm(p=2, dim=-1, keepdim=True)

        image_embeddings_dino.append(embeds_dino.to(dtype=torch.bfloat16).detach().cpu())
        image_embeddings_siglip.append(embeds_siglip.to(dtype=torch.bfloat16).detach().cpu())

        del processed, out, embeds_dino, embeds_siglip, image_summary_dict, image_summary_dinov3, image_summary_siglip2, patch_tokens_list

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


def compute_similarity_matrix_chunked(image_embeddings, text_embeddings, device="cuda", batch_size=1024):
    """Compute similarity matrix in chunks to avoid OOM or slow CPU matmul."""
    print("Computing similarity matrix (chunked)...")
    num_images = image_embeddings.shape[0]
    num_texts = text_embeddings.shape[0]
    
    # Pre-allocate result on CPU
    sim_matrix = torch.empty((num_images, num_texts), dtype=image_embeddings.dtype)
    
    # Ensure embeddings are on CPU to start (they should be)
    image_embeddings = image_embeddings.cpu()
    text_embeddings = text_embeddings.cpu()

    # Move text embeddings to GPU once if they fit, otherwise chunk both
    # Text embeddings are usually smaller (150k * 1024 * 2 bytes = 300MB) -> fits easily
    try:
        text_embeddings_gpu = text_embeddings.to(device, non_blocking=True)
        
        for i in tqdm(range(0, num_images, batch_size), desc="Sim matrix chunks"):
            end = min(i + batch_size, num_images)
            img_chunk = image_embeddings[i:end].to(device, non_blocking=True)
            
            # Compute chunk
            sim_chunk = torch.matmul(img_chunk, text_embeddings_gpu.T)
            
            # Move back to CPU
            sim_matrix[i:end] = sim_chunk.cpu()
            
            del sim_chunk, img_chunk
        del text_embeddings_gpu
        torch.cuda.empty_cache()
        
    except RuntimeError: # OOM
        print("OOM with full text matrix on GPU, falling back to double chunking...")
        torch.cuda.empty_cache()
        for i in tqdm(range(0, num_images, batch_size), desc="Sim matrix chunks (rows)"):
            end_i = min(i + batch_size, num_images)
            img_chunk = image_embeddings[i:end_i].to(device)
            
            for j in range(0, num_texts, batch_size):
                end_j = min(j + batch_size, num_texts)
                txt_chunk = text_embeddings[j:end_j].to(device)
                
                sim_chunk = torch.matmul(img_chunk, txt_chunk.T)
                sim_matrix[i:end_i, j:end_j] = sim_chunk.cpu()
                del txt_chunk, sim_chunk
            del img_chunk
            
    return sim_matrix


def compute_text_embeddings_siglip2(
    texts: list[str],
    model_name: str,
    device: str,
    batch_size: int = 1024
):
    """Compute normalized text embeddings using SigLIP2."""
    text_tower = AutoModel.from_pretrained(model_name).text_model.to(device)
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
            embeds_chunks.append(out)
            
    del text_tower, processor
    torch.cuda.empty_cache()
    return torch.cat(embeds_chunks, dim=0)


def average_embeddings_over_templates(embeddings, num_classes, num_templates):
    """Reshape [C*T, D] -> [C, T, D] -> mean -> [C, D], then normalize."""
    embeddings = embeddings.reshape(num_classes, num_templates, -1)
    embeddings = embeddings.mean(dim=1)
    embeddings = embeddings / embeddings.norm(p=2, dim=-1, keepdim=True)
    return embeddings


def compute_retrieval_metrics_from_similarity(similarity_matrix: torch.Tensor, image_to_texts_map, device=None):
    """
    Compute bidirectional retrieval metrics given a precomputed similarity (logits) matrix.
    Vectorized implementation for speed.
    """
    # Auto-detect device if not provided, preferring GPU if available
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Ensure matrix is on device for fast ops
    if similarity_matrix.device.type == "cpu":
        similarity_matrix = similarity_matrix.to(device)
    
    num_images, num_texts = similarity_matrix.shape
    
    # -------------------------------------------------------------------------
    # 1. Text-to-Image (T2I)
    # -------------------------------------------------------------------------
    print("Computing Text-to-Image retrieval metrics ...")
    
    # Build ground truth mapping: text_idx -> correct_image_idx
    # This replaces the slow linear search in the loop
    text_to_image = torch.zeros(num_texts, dtype=torch.long, device=device)
    for img_idx, text_indices in image_to_texts_map.items():
        for t_idx in text_indices:
            text_to_image[t_idx] = img_idx

    # Transpose so rows are queries (texts) and columns are targets (images)
    sims_t2i = similarity_matrix.T 
    
    # Get the score of the ground truth image for each text
    gt_indices = text_to_image.unsqueeze(1) # [num_texts, 1]
    gt_scores = sims_t2i.gather(1, gt_indices) # [num_texts, 1]
    
    # Compute rank: count how many images have a score strictly greater than the GT
    # This is much faster than full argsort
    # Ranks are 1-based
    ranks = (sims_t2i > gt_scores).sum(dim=1) + 1
    ranks = ranks.float()

    t2i_recalls = {
        1: (ranks <= 1).float().mean().item() * 100,
        5: (ranks <= 5).float().mean().item() * 100,
        10: (ranks <= 10).float().mean().item() * 100
    }

    # -------------------------------------------------------------------------
    # 2. Image-to-Text (I2T)
    # -------------------------------------------------------------------------
    print("Computing Image-to-Text retrieval metrics ...")
    
    # For I2T, we just need to check if ANY correct caption is in the top K.
    # Since K is small (10), topk is efficient.
    k_max = 10
    _, topk_indices = torch.topk(similarity_matrix, k=k_max, dim=1) # [num_images, 10]
    
    # Build a dense ground truth mask: [num_images, num_texts]
    gt_mask = torch.zeros((num_images, num_texts), dtype=torch.bool, device=device)
    for img_idx, text_indices in image_to_texts_map.items():
        gt_mask[img_idx, text_indices] = True
        
    # Check hits: Gather mask values at the predicted top-k indices
    hits = torch.gather(gt_mask, 1, topk_indices) # [num_images, 10]
    
    i2t_recalls = {}
    for k in [1, 5, 10]:
        # Check if there is at least one True in the top k results
        i2t_recalls[k] = hits[:, :k].any(dim=1).float().mean().item() * 100
        
    return t2i_recalls, i2t_recalls
