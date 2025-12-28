# PCA visualization for Falcon Vision standalone model
import argparse
import os
import pickle
import random
import glob
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch._inductor.config
torch._inductor.config.triton.unique_kernel_names = True
torch._dynamo.config.allow_unspec_int_on_nn_module = True
torch._inductor.config.triton.cudagraph_skip_dynamic_graphs = True
torch._inductor.config.fx_graph_cache = True
torch._functorch.config.enable_autograd_cache = True
torch.set_float32_matmul_precision("high")

from sklearn.decomposition import PCA

# Standalone model imports
from amoe import load_amoe_model


def load_image(path: str) -> Image.Image:
    img = Image.open(path).convert("RGB")
    print(f"Image size: {img.size}")
    return img


@torch.inference_mode()
def extract_patch_features(
    model,
    processor, 
    images: List[Image.Image],
    device: str = "cuda",
    max_num_patches: int = 256,
):
    """Extract patch features from images using the standalone model."""
    features_per_image = []
    
    for image in images:
        # Use the proper image processor
        processed = processor(
            image,
            max_num_patches=max_num_patches,
            n_storage_tokens=model.n_storage_tokens,
            pad=False,
        )
        
        pixel_values = processed["pixel_values"].to(device, dtype=model.dtype)
        padding_mask = processed["padding_mask"].to(device)
        spatial_shapes = processed["spatial_shape"].to(device)
        
        H, W = spatial_shapes[0].tolist()
        print(f"Spatial shapes: H={H}, W={W}, total_patches={H*W}")
        print(f"Pixel values: {pixel_values.shape}")
        out = model(
            pixel_values=pixel_values,
            padding_mask=padding_mask,
            spatial_shapes=spatial_shapes,
            compile=True,
        )
        
        patch_feats = out["patch_features"]
        
        # Features are (N, L, D) - squeeze batch dimension
        feats_siglip = patch_feats["siglip2"].squeeze(0)  # (L, D)
        feats_dinov3 = patch_feats["dinov3"].squeeze(0)  # (L, D)
        feats_amoe = patch_feats["amoe"].squeeze(0)  # (L, D)
        
        features_per_image.append({
            "features_siglip": feats_siglip,
            "features_dinov3": feats_dinov3,
            "features_amoe": feats_amoe,
            "grid_hw": (H, W),
        })
    
    return features_per_image


def fit_and_project_pca(feats_2d: torch.Tensor, n_components: int = 3, whiten: bool = True) -> np.ndarray:
    x = feats_2d.detach().float().cpu().numpy()
    pca = PCA(n_components=n_components, whiten=whiten)
    pca.fit(x)
    proj = pca.transform(x)
    return proj


def render_pca_image(
    image_rgb: Image.Image,
    projected_L3: Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]],
    grid_hw: tuple,
    save_path: str,
    title: Optional[str] = None,
):
    """Render PCA visualizations for available feature types."""
    projected_amoe, projected_siglip, projected_dinov3 = projected_L3

    H, W = grid_hw

    def create_pca_grid(projected_features):
       
        grid_hw3 = projected_features.reshape(H, W, 3).astype(np.float32)
        # Sigmoid(2x) for vibrant colors
        grid_hw3 = 1.0 / (1.0 + np.exp(-2.0 * grid_hw3))

        return grid_hw3

    viz_items = []
    if projected_siglip is not None:
        viz_items.append(("SigLIP PCA", create_pca_grid(projected_siglip)))
    if projected_dinov3 is not None:
        viz_items.append(("DINO v3 PCA", create_pca_grid(projected_dinov3)))
    if projected_amoe is not None:
        viz_items.append(("AMOE PCA", create_pca_grid(projected_amoe)))

    n_cols = max(2, len(viz_items))
    plt.figure(figsize=(4 * n_cols, 8), dpi=200)

    top_col = (n_cols + 1) // 2
    for c in range(1, n_cols + 1):
        plt.subplot(2, n_cols, c)
        if c == top_col:
            plt.imshow(image_rgb)
            plt.axis("off")
            plt.title("Original Image")
        else:
            plt.axis("off")

    for idx, (name, grid) in enumerate(viz_items, start=1):
        plt.subplot(2, n_cols, n_cols + idx)
        plt.imshow(grid)
        plt.axis("off")
        plt.title(name)

    if title:
        plt.suptitle(title, fontsize=14)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def load_model_and_processor(
    ckpt_path: str,
    config_name: str,
    device: str = "cuda",
    min_pixels: int = 128 * 128,
    max_pixels: int = 256 * 256,
):
    """Load the standalone Falcon Vision model and image processor."""
    # Update config with register count if it exists

    
    print(f"Loading model with config: {config_name}")
    
    # Load model using the package function
    model, processor = load_amoe_model(
        checkpoint_path=ckpt_path,
        config_name=config_name,
        device=device,
        max_pixels=max_pixels,
    )
    
    # Ensure model is in bfloat16
    model = model.to(torch.bfloat16)
    
    return model, processor


def sample_jpg_images(input_dir: str, num_samples: int = 10) -> List[str]:
    """Sample JPG images from input directory."""
    jpg_pattern = os.path.join(input_dir, "*.jpg")
    jpg_files = glob.glob(jpg_pattern)
    
    # Also try PNG
    png_pattern = os.path.join(input_dir, "*.png")
    jpg_files.extend(glob.glob(png_pattern))

    if len(jpg_files) == 0:
        raise ValueError(f"No JPG/PNG files found in {input_dir}")

    random.seed(42)
    num_to_sample = min(num_samples, len(jpg_files))
    sampled_files = random.sample(jpg_files, num_to_sample)

    print(f"Found {len(jpg_files)} image files, sampling {num_to_sample} images")
    return sampled_files


def process_single_image(
    image_path: str,
    output_dir: str,
    model,
    processor,
    device: str,
    max_num_patches: int = 256,
) -> None:
    """Process a single image and save the visualization."""
    image = load_image(image_path)

    features_info = extract_patch_features(
        model=model,
        processor=processor,
        images=[image],
        device=device,
        max_num_patches=max_num_patches,
    )
    info = features_info[0]
    H, W = info["grid_hw"]
    num_valid = H * W
    
    # Extract only valid (non-padding) features
    feats_LD_siglip = info["features_siglip"][:num_valid]
    feats_LD_dinov3 = info["features_dinov3"][:num_valid]
    feats_LD_amoe = info["features_amoe"][:num_valid]
    
    print(f"Feature shapes (valid only) - siglip: {feats_LD_siglip.shape}, dinov3: {feats_LD_dinov3.shape}, amoe: {feats_LD_amoe.shape}")


    projected_all_siglip = fit_and_project_pca(feats_LD_siglip)
    projected_all_dinov3 = fit_and_project_pca(feats_LD_dinov3)
    projected_all_amoe = fit_and_project_pca(feats_LD_amoe)
    
    image_basename = os.path.splitext(os.path.basename(image_path))[0]
    output_filename = f"{image_basename}_pca_vis.png"
    output_path = os.path.join(output_dir, output_filename)

    print(f"Projected shapes - amoe: {projected_all_amoe.shape}, siglip: {projected_all_siglip.shape}, dinov3: {projected_all_dinov3.shape}")
    render_pca_image(
        image_rgb=image,
        projected_L3=(projected_all_amoe, projected_all_siglip, projected_all_dinov3),
        grid_hw=info["grid_hw"],
        save_path=output_path,
        title=os.path.basename(image_path),
    )

    print(f"Saved visualization: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize PCA of AMOE patch features")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing images")
    parser.add_argument("--output_path", type=str, required=True, help="Base output directory")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of images to sample")
    parser.add_argument("--config_name", type=str, default="18-layers-distillation")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_num_patches", type=int, default=256)
    args = parser.parse_args()

    output_dir = os.path.join(args.output_path, "pca_visualizations")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    sampled_images = sample_jpg_images(args.input_dir, args.num_samples)

    print("Loading model and processor...")
    model, processor = load_model_and_processor(
        ckpt_path=args.ckpt_path,
        config_name=args.config_name,
        device=args.device,
        min_pixels=128*128,
        max_pixels=(args.max_num_patches**0.5*16)**2,
    )


    print(f"Processing {len(sampled_images)} images...")
    for i, image_path in enumerate(sampled_images, 1):
        print(f"Processing image {i}/{len(sampled_images)}: {os.path.basename(image_path)}")
        process_single_image(
            image_path=image_path,
            output_dir=output_dir,
            model=model,
            processor=processor,
            device=args.device,
            max_num_patches=args.max_num_patches,
        )

    print(f"Completed! All visualizations saved in: {output_dir}")


if __name__ == "__main__":
    main()
