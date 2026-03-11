# AMoE: Agglomerative Mixture-of-Experts Vision Foundation Models

**Accepted at CVPR 2026**

[![Project Website](https://img.shields.io/badge/Project-Website-blue)](https://sofianchay.github.io/amoe/)
[![arXiv](https://img.shields.io/badge/arXiv-2512.20157-b31b1b.svg)](https://arxiv.org/abs/2512.20157)
[![Hugging Face Models](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-yellow)](https://huggingface.co/tiiuae/amoe)
[![Hugging Face Datasets](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset(soon)-green)]()

A vision encoder distilled from DINOv3 and SigLIP2 teachers, supporting multi-resolution image understanding with Mixture-of-Experts (MoE) architecture.

![Main fig](./main_fig.png)

## Model Zoo

| Model | Architecture | Active Params | Total Params | Config Name | Checkpoint |
|-------|-------------|--------------|-------------|-------------|------------|
| AMoE | MoE (top-6/28) | 0.15B | 0.6B | `18-layers-distillation` | [Download](https://github.com/tiiuae/amoe/releases/download/AMoE-checkpoint/amoe_ckpt_paper_version.pt) |
| AMoE-Ultrasparse | MoE (top-2/28) | 0.15B | 0.6B | `ultrasparse` | [Download](https://github.com/tiiuae/amoe/releases/download/AMoE-checkpoint/ultrasparse.pt) |
| AMoE-Dense-L | Dense | 0.6B | 0.6B | `dense-L` | [Download](https://github.com/tiiuae/amoe/releases/download/AMoE-checkpoint/dense-L.pt) |
| AMoE-Dense-S | Dense | 0.07B | 0.07B | `dense-S` | [Download](https://github.com/tiiuae/amoe/releases/download/AMoE-checkpoint/dense-S.pt) |
| AMoE-Dense-XS | Dense | 0.03B | 0.03B | `dense-XS` | [Download](https://github.com/tiiuae/amoe/releases/download/AMoE-checkpoint/dense-XS.pt) |

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

## Quick Start

### AMoE-Ultrasparse (MoE, 0.15B active / 0.6B total)

```python
from amoe import load_amoe_model
from PIL import Image
import torch

# Download checkpoint
# wget https://github.com/tiiuae/amoe/releases/download/AMoE-checkpoint/ultrasparse.pt

model, processor = load_amoe_model(
    checkpoint_path="ultrasparse.pt",
    config_name="ultrasparse",
    device="cuda",
)
model = model.to(torch.bfloat16)

image = Image.open("image.jpg")
inputs = processor(image, return_tensors="pt")

with torch.no_grad():
    pixel_values = inputs["pixel_values"].to("cuda", dtype=torch.bfloat16)
    spatial_shapes = inputs["spatial_shape"].to("cuda")
    padding_mask = inputs["padding_mask"].to("cuda")

    outputs = model(
        pixel_values=pixel_values,
        spatial_shapes=spatial_shapes,
        padding_mask=padding_mask
    )

    # DINOv3-style patch features
    patch_features = outputs["patch_features"]["dinov3"]  # (N, L, 1024)

    # SigLIP2-style pooled features
    pooled_features = outputs["summary_features"]["siglip2"]  # (N, 1152)

    # Native model features
    amoe_features = outputs["patch_features"]["amoe"]  # (N, L, 768)
```

### AMoE-Dense-L (Dense, 0.6B)

```python
from amoe import load_amoe_model
from PIL import Image
import torch

# Download checkpoint
# wget https://github.com/tiiuae/amoe/releases/download/AMoE-checkpoint/dense-L.pt

model, processor = load_amoe_model(
    checkpoint_path="dense-L.pt",
    config_name="dense-L",
    device="cuda",
)
model = model.to(torch.bfloat16)

image = Image.open("image.jpg")
inputs = processor(image, return_tensors="pt")

with torch.no_grad():
    pixel_values = inputs["pixel_values"].to("cuda", dtype=torch.bfloat16)
    spatial_shapes = inputs["spatial_shape"].to("cuda")
    padding_mask = inputs["padding_mask"].to("cuda")

    outputs = model(
        pixel_values=pixel_values,
        spatial_shapes=spatial_shapes,
        padding_mask=padding_mask
    )

    # DINOv3-style patch features
    patch_features = outputs["patch_features"]["dinov3"]  # (N, L, 1024)

    # SigLIP2-style pooled features
    pooled_features = outputs["summary_features"]["siglip2"]  # (N, 1152)

    # Native model features
    amoe_features = outputs["patch_features"]["amoe"]  # (N, L, 1280)
```

## PCA Visualization

To visualize the principal components of the features:

```bash
python pca_maps.py \
    --ckpt_path path/to/checkpoint.pt \
    --input_dir path/to/images/ \
    --output_path ./output_viz/ \
    --num_samples 10
```

Sample output:

![PCA visualization sample 1](pca_maps_amoe/pca_visualizations/pca_instance.png)

## HF usage

```python
import torch
from PIL import Image
from transformers import AutoModel, AutoImageProcessor

# Load model and processor
model_id = "tiiuae/amoe"
model = AutoModel.from_pretrained(model_id, trust_remote_code=True).to("cuda", dtype=torch.bfloat16)
processor = AutoImageProcessor.from_pretrained(model_id, trust_remote_code=True)

# Preprocess image
image = Image.open("image.jpg").convert("RGB")
inputs = processor(image, return_tensors="pt").to("cuda")
inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

# Inference
with torch.no_grad():
    outputs = model(**inputs)

# Access specialized features
# Options: 'amoe' (768d), 'siglip2' (1152d), 'dinov3' (1024d)
patch_features = outputs["patch_features"]["amoe"]    # (Batch, Tokens, 768)
summary_features = outputs["summary_features"]["siglip2"] # (Batch, 1152)

```
## Citation

If you use AMoE in your research, please cite:

```bibtex
@article{chaybouti2025amoe,
  title={AMOE: Agglomerative Mixture-of-Experts Vision Foundation Models},
  author={Chaybouti, Sofian and Narayan, Sanath and Dahou, Yasser and Le Khac, Phuc H. and Singh, Ankit and Huynh, Ngoc Dung and Para, Wamiq Reyaz and Kuehne, Hilde and Hacid, Hakim},
  journal={arXiv preprint arXiv:2512.20157},
  year={2025}
}
```
