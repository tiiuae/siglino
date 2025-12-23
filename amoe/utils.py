# Utilities for Falcon Vision
# Model loading and image preprocessing without tokenizer dependency

import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from PIL import Image
from typing import Union, List
import os

from .model import AMOE
from .configs import AMOEArgs, amoe_configs
from .image_processor import AMOEImageProcessor



def load_amoe_model(
    checkpoint_path: str,
    config_name: str = "18-layers-distillation",
    device: Union[str, torch.device] = "cuda",
    dtype: torch.dtype | None = None,
    **kwargs,
) -> tuple[AMOE, AMOEImageProcessor]:
    """
    Load a AMOE model from a checkpoint.
    
    Args:
        checkpoint_path: Path to the model checkpoint
        config_name: Name of the model configuration
        device: Device to load the model on
        dtype: Optional dtype to cast model weights to (e.g. torch.bfloat16)
    
    Returns:
        Tuple of (model, image_processor)
    """
    # Get configuration
    if config_name in amoe_configs:
        args = amoe_configs[config_name]
    else:
        raise ValueError(f"Unknown config: {config_name}. Available: {list(amoe_configs.keys())}")
    
    # Create model
    model = AMOE(args)
    
    # Standard PyTorch checkpoint
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        
    model.load_state_dict(state_dict)
    
    if dtype is None:
        model = model.to(device=device)
    else:
        model = model.to(device=device, dtype=dtype)
    model.eval()
    
    # Create image processor
    image_processor = AMOEImageProcessor(patch_size=args.spatial_patch_size, **kwargs)
    
    return model, image_processor


# def convert_torchtitan_checkpoint(
#     torchtitan_ckpt_path: str,
#     output_path: str,
#     config_name: str = "0.25B-1B-a-tall-se-24l16e-route-distillation",
# ):
#     """
#     Convert a torchtitan checkpoint to standalone format.
#     
#     This handles the key mapping differences between the torchtitan
#     DistillPerceptionTransformerMultiTeacher and FalconVisionEncoder.
#     """
#     # Load torchtitan checkpoint
#     if os.path.isdir(torchtitan_ckpt_path):
#         from torch.distributed.checkpoint import load as dcp_load
#         config = omni_falcon_perception_configs[config_name]
#         config.max_seq_len = 2048
#         config.seq_len = 2304 + 5
#         config.vocab_size = 65536 
#         config.eos_id = 31999
#         config.dtype = torch.bfloat16
#         config.use_grouped_mm = False
#         config.use_flex_attn = True
#         config.attn_mask_type = "distill_mask"
#         config.img_start_id = 31998
#         config.img_end_id = 31997
#         config.img_id = 31996
#         config.eager = True
#         config.n_storage_tokens = 4
#         config.img_row_sep_id = 31995
#         config.vid_start_id = 31994
#         config.vid_end_id = 31993
#         config.frame_sep_id = 31992
#         config.image_mask_token_id = 31991
#         config.image_cls_token_id = 31990
#         config.image_reg_1_token_id = 31989
#         config.image_reg_2_token_id = 31988
#         config.image_reg_3_token_id = 31987
#         config.image_reg_4_token_id = 31986
#         config.cls_weight = 0
#         config.patch_weight = 0
#         config.storage_weight = 0
#         config.pairwise_distance_weight = 0
#         config.pairwise_cosine_weight = 0
#         config.pairwise_distance_patch_weight = 0
#         config.pairwise_cosine_patch_weight = 0
#         config.high_res_distillation_weight = 0
#         config.teachers = ("siglip2", "dinov3")
#         config.teachers_dim = (1152, 1024)
#         config.optimizable_teachers = ("siglip2", "dinov3")
#         config.average_patch_loss = False
#         config.weighted_patch_loss = False
#         config.jitter_rope = False
#         config.use_phis = False
#         config.use_pixel_head = True
# 
#         # Load model
#         model = DistillPerceptionTransformerMultiTeacher(config).to("cuda")
#         state_dict = model.state_dict()
#         state_dict.pop('freqs_cis', None)
#         keys = list(state_dict.keys())
#         for k in keys:
#             if "coord" in k:
#                 state_dict.pop(k, None)
#             if "size" in k:
#                 state_dict.pop(k, None)
#             if "proj_segm" in k:
#                 state_dict.pop(k, None)
#             if "itok_upsampler" in k:
#                 state_dict.pop(k, None)
#             if "rope_upsampler" in k:
#                 state_dict.pop(k, None)
#         
#         dcp_load(state_dict, checkpoint_id=torchtitan_ckpt_path)
#     else:
#         state_dict = torch.load(torchtitan_ckpt_path, map_location="cpu", weights_only=False)
#         if "model" in state_dict:
#             state_dict = state_dict["model"]
#     
#     # Key mapping from torchtitan to standalone
#     key_map = {
#         "tok_embeddings": None,  # Remove text embeddings
#         "output": None,  # Remove text output
#         "pixel_mlp": None,  # Remove pixel head
#         "proj_segm": None,  # Remove segmentation head
#         "itok_upsampler": None,  # Remove upsampler
#         "coord_encoder": None,  # Remove coordinate heads
#         "coord_decoder": None,
#         "size_encoder": None,
#         "size_decoder": None,
#         "phis_statistics": None,  # Remove PHIs statistics
#         "rope_upsampler": None,  # Remove RoPE upsampler
#     }
#     
#     new_state_dict = {}
#     for k, v in state_dict.items():
#         # Skip keys that should be removed
#         skip = False
#         for prefix in key_map.keys():
#             if k.startswith(prefix) or k.startswith(f"model.{prefix}"):
#                 skip = True
#                 break
#         if skip:
#             continue
#         
#         # Remove "model." prefix if present
#         new_key = k[6:] if k.startswith("model.") else k
#         print(new_key)
#         new_state_dict[new_key] = v
#     
#     # Save converted checkpoint
#     torch.save(new_state_dict, output_path)
#     print(f"Saved converted checkpoint to {output_path}")


# Feature dimension constants
FEATURE_DIM_DICT = {
    "dinov3": 1024,
    "siglip2": 1152,
    "amoe": 768,  # Model dimension
}

PATCH_SIZE = 16


