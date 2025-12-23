# Segmentation utilities for Falcon Vision evaluation
# Uses the standalone falcon_vision model

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from amoe import AMOE
from amoe.utils import FEATURE_DIM_DICT, PATCH_SIZE, load_amoe_model


def build_backbone_and_processor(
    ckpt_path: str,
    device: torch.device = torch.device("cuda"),
    feature_type: str = "dinov3",
    dtype: torch.dtype = torch.bfloat16,
):
    """
    Build the backbone model and image processor.
    
    Args:
        ckpt_path: Path to model checkpoint
        device: Device to load model on
        feature_type: Type of features to extract
        image_size: Image size for processing
    
    Returns:
        Tuple of (backbone, image_processor)
    """
    # Load model
    model, image_processor = load_amoe_model(
        checkpoint_path=ckpt_path,
        device=device,
        dtype=dtype,
        do_resize=False
    )
    
    # Wrap in backbone interface
    backbone = AMOEBackbone(model, feature_type=feature_type)
    
    return backbone, image_processor


class AMOEBackbone(nn.Module):
    """Wrapper that provides a unified interface for feature extraction."""
    
    def __init__(self, model: AMOE, feature_type: str = "dinov3"):
        super().__init__()
        self.model = model
        self.feature_type = feature_type
        self.patch_size = model.patch_size
    
    def forward(
        self,
        pixel_values: torch.Tensor,
        spatial_shapes: torch.Tensor | None = None,
        compile: bool = True,
    ) -> dict:
        """
        Extract features from images.
        
        Args:
            pixel_values: Preprocessed image patches (N, L, C*P*P)
            spatial_shape: Patch grid shape per image (N, 2)
            compile: Whether to use compiled attention
        
        Returns:
            Dictionary with output features
        """
        with torch.no_grad():
            outputs = self.model(
                pixel_values=pixel_values,
                spatial_shapes=spatial_shapes,
                compile=compile,
            )
        return outputs


class AMOELinearSeg(nn.Module):
    """Linear segmentation head on top of Falcon Vision backbone."""
    
    def __init__(
        self,
        backbone: AMOEBackbone,
        num_classes: int = 21,
        feature_dim: int = 1024,
        feature_type: str = "dinov3",
        image_size: int = 256,
    ):
        super().__init__()
        self.backbone = backbone
        self.feature_type = feature_type
        self.image_size = image_size
        self.patch_size = backbone.patch_size
        
        # Linear segmentation head
        self.head = nn.Conv2d(feature_dim, num_classes, kernel_size=1)
    
    def forward(
        self,
        pixel_values: torch.Tensor,
        spatial_shape: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for segmentation.
        
        Args:
            pixel_values: Image patches (N, L, C)
            spatial_shape: Spatial shape (N, 2)
        
        Returns:
            Segmentation logits (N, num_classes, H, W)
        """
        # Extract features
        outputs = self.backbone(
            pixel_values=pixel_values,
            spatial_shapes=spatial_shape,
            compile=True,
        )
        
        # Get patch features for the desired type
        feats = outputs["output"][self.feature_type]  # (N, L, D)
        
        N, L, D = feats.shape
        H_patch = int(spatial_shape[0, 0].item())
        W_patch = int(spatial_shape[0, 1].item())
        
        # Reshape to spatial grid
        feats = feats.view(N, H_patch, W_patch, D)
        feats = feats.permute(0, 3, 1, 2)  # (N, D, H_patch, W_patch)
        
        # Apply segmentation head
        logits = self.head(feats)  # (N, C, H_patch, W_patch)
        
        # Upsample to image size
        logits = F.interpolate(
            logits,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )
        
        return logits
    
    def forward_from_precomputed(self, feats: torch.Tensor) -> torch.Tensor:
        """Forward using precomputed features."""
        logits = self.head(feats)
        logits = F.interpolate(
            logits,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )
        return logits


def make_collate_fn(
    image_processor,
    max_num_patches: int = 256,
    output_dtype: torch.dtype = torch.bfloat16,
):
    """Create a collate function for the dataloader."""
    
    def collate_fn(batch):
        images, masks = zip(*batch)
        
        # Process images (patchify + pad + mask)
        enc = image_processor(
            list(images),
            max_num_patches=max_num_patches,
            pad=True,
            output_dtype=output_dtype,
            mask_dtype=output_dtype,
        )
        
        # Stack masks
        masks = torch.stack(masks)
        
        return {
            "pixel_values": enc["pixel_values"],
            "spatial_shape": enc["spatial_shape"],
            "targets": masks,
        }
    
    return collate_fn

