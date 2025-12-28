import torch
import torch.distributed as dist
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
import sys
import os

from amoe import load_amoe_model

def make_collate_fn(image_processor, max_num_patches: int):
    def _collate(batch):
        batch = [item for item in batch if item is not None]
        if not batch:
            return None

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

@torch.no_grad()
def process_preprocessed_batch(batch, model, device):
    pixel_values = batch["pixel_values"].to(device, non_blocking=True, dtype=torch.bfloat16)
    padding_mask = batch["padding_mask"].to(device, non_blocking=True)
    spatial_shapes = batch["spatial_shapes"].to(device, non_blocking=True)

    outputs = model(
        pixel_values=pixel_values,
        padding_mask=padding_mask,
        spatial_shapes=spatial_shapes,
    )

    summaries = outputs["summary_features"]  # {"dinov3": (B, D_dino), "siglip2": (B, D_siglip)}

    return {
        'summaries': summaries,
        'class_ids': batch.get('class_ids'),
        'categories': batch.get('categories'),
        'image_paths': batch.get('image_paths'),
    }

# -----------------------------------------------------------------------------
# kNN Specific Utilities
# -----------------------------------------------------------------------------

def class_votes(sim: torch.Tensor, labels: torch.Tensor, num_classes: int, temperature: float):
    weights = torch.exp(sim / temperature)  # [B, K]
    cls_vec = torch.zeros(weights.shape[0], num_classes, dtype=weights.dtype, device=weights.device)
    cls_vec.scatter_add_(dim=1, index=labels, src=weights)
    return cls_vec  # [B, C]

def fuse_votes_entropy_weighted(v1: torch.Tensor, v2: torch.Tensor, beta: float = 5.0):
    # Normalize to probabilities for confidence estimation
    p1 = v1 / v1.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    p2 = v2 / v2.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    H1 = -(p1.clamp_min(1e-8) * p1.clamp_min(1e-8).log()).sum(dim=-1, keepdim=True)
    H2 = -(p2.clamp_min(1e-8) * p2.clamp_min(1e-8).log()).sum(dim=-1, keepdim=True)
    w1 = torch.softmax(torch.cat([-beta * H1, -beta * H2], dim=-1), dim=-1)[..., 0:1]
    w2 = 1.0 - w1
    return w1 * v1 + w2 * v2

def _pad(tensor: torch.Tensor, dim0: int):
    valid_mask = torch.ones(dim0, dtype=torch.bool, device=tensor.device)
    valid_mask[tensor.shape[0]:].fill_(False)
    if tensor.shape[0] == dim0:
        return tensor, valid_mask
    ret = torch.empty(dim0, *tensor.shape[1:], dtype=tensor.dtype, device=tensor.device)
    ret[:tensor.shape[0]].copy_(tensor)
    return ret, valid_mask

def _all_to_all(t: torch.Tensor):
    input_tensors = list(t)
    output_tensors = [torch.empty_like(v) for v in input_tensors]
    dist.all_to_all(output_tensors, input_tensors)
    return torch.stack(output_tensors)

def _distributed_topk(
    queries: torch.Tensor,
    keys: torch.Tensor,
    labels: torch.Tensor,
    K: int,
    distributed: bool,
):
    if distributed:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        max_queries = torch.tensor(queries.shape[0], dtype=torch.int64, device=queries.device)
        dist.all_reduce(max_queries, dist.ReduceOp.MAX)
        max_queries = max_queries.item()
        queries, valid_mask = _pad(queries, max_queries)

        all_queries = torch.empty(
            world_size, queries.shape[0], queries.shape[1],
            dtype=queries.dtype, device=queries.device
        )
        dist.all_gather_into_tensor(all_queries, queries)
    else:
        all_queries = queries.unsqueeze(0)
        valid_mask = torch.ones(queries.shape[0], dtype=torch.bool, device=queries.device)

    # keys: (N_train, D)
    # all_queries: (W, Q, D)
    # We want (W, Q, K) top-k indices from keys
    
    # Chunked computation to avoid OOM if keys are large
    W, Q, D = all_queries.shape
    all_queries_flat = all_queries.view(W * Q, D)
    
    # Assuming keys fit in memory, otherwise we need to chunk keys too.
    # similarity = torch.matmul(all_queries, keys.T)  # W,Q,N_train
    
    chunk_size = 1024
    num_queries = W * Q
    max_sim_list = []
    max_idxs_list = []
    
    for i in range(0, num_queries, chunk_size):
        q_chunk = all_queries_flat[i : i + chunk_size]
        sim_chunk = torch.matmul(q_chunk, keys.T) # (chunk, N_train)
        vals, idxs = torch.topk(sim_chunk, k=K, dim=1, largest=True, sorted=False)
        max_sim_list.append(vals)
        max_idxs_list.append(idxs)
        
    max_sim = torch.cat(max_sim_list, dim=0).view(W, Q, K)
    max_idxs = torch.cat(max_idxs_list, dim=0).view(W, Q, K)

    max_labels = labels[max_idxs.flatten()].reshape_as(max_idxs)

    if distributed:
        max_sim = _all_to_all(max_sim)
        max_labels = _all_to_all(max_labels)

    max_sim = max_sim.permute(1, 2, 0).flatten(1)      # N,K*W
    max_labels = max_labels.permute(1, 2, 0).flatten(1)

    if distributed:
        max_sim, max_idxs2 = torch.topk(max_sim, k=K, dim=1, largest=True, sorted=False)
        max_labels = torch.gather(max_labels, dim=1, index=max_idxs2)

    max_sim = max_sim[valid_mask]
    max_labels = max_labels[valid_mask]
    return max_sim, max_labels

def build_embeddings(
    dataloader: DataLoader,
    model,
    device,
):
    embeddings_dinov3 = []
    embeddings_siglip2 = []
    labels = []

    for batch in tqdm(dataloader, desc="Building embeddings"):
        if batch is None:
            continue
        out = process_preprocessed_batch(batch, model, device)

        feats_dinov3 = out['summaries']['dinov3']
        feats_siglip2 = out['summaries']['siglip2']

        feats_dinov3 = feats_dinov3 / feats_dinov3.norm(p=2, dim=-1, keepdim=True)
        feats_siglip2 = feats_siglip2 / feats_siglip2.norm(p=2, dim=-1, keepdim=True)
        embeddings_dinov3.append(feats_dinov3.cpu())
        embeddings_siglip2.append(feats_siglip2.cpu())
        labels.append(out['class_ids'])

    return torch.cat(embeddings_dinov3, dim=0).to(device), torch.cat(embeddings_siglip2, dim=0).to(device), torch.cat(labels, dim=0).to(device)
