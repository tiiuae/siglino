# SigLino: Vision Foundation Models distilled from SigLIP2 and DINOv3

**Accepted at CVPR 2026**

[![Project Website](https://img.shields.io/badge/Project-Website-blue)](https://sofianchay.github.io/siglino/)
[![arXiv](https://img.shields.io/badge/arXiv-2512.20157-b31b1b.svg)](https://arxiv.org/abs/2512.20157)
[![Hugging Face Models](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-yellow)](https://huggingface.co/collections/tiiuae/siglino-vision-foundation-models)
[![Hugging Face Datasets](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset(soon)-green)]()

This work stems from the **CVPR 2026 AMoE paper**, which designs and applies distillation into a Mixture-of-Experts (MoE) vision architecture. We have chosen the name **SigLino** for better clarity (SigLIP2 + DINOv3).

SigLino is a vision encoder distilled from DINOv3 and SigLIP2 teachers, supporting multi-resolution image understanding with a Mixture-of-Experts (MoE) architecture.

<p align="center">
  <img src="main_fig.png" width="800">
</p>

## Model Zoo


| Model           | Architecture   | Active Params | Total Params | Config Name           | Checkpoint                                                                                      |
| --------------- | -------------- | ------------- | ------------ | --------------------- | ----------------------------------------------------------------------------------------------- |
| SigLino-MoE-0.3-0.6B | MoE (top-6/28) | 0.3B          | 0.6B         | `siglino-moe-0.3B`    | [Download](https://github.com/tiiuae/amoe/releases/download/AMoE-checkpoint/amoe-0.3B.pt)   |
| SigLino-MoE-0.15-0.6B | MoE (top-2/28) | 0.15B         | 0.6B         | `siglino-moe-0.15B`   | [Download](https://github.com/tiiuae/amoe/releases/download/AMoE-checkpoint/amoe-0.15B.pt)  |
| SigLino-0.6B    | Dense          | 0.6B          | 0.6B         | `siglino-0.6B`        | [Download](https://github.com/tiiuae/amoe/releases/download/AMoE-checkpoint/dense-0.6B.pt)       |
| SigLino-70M     | Dense          | 0.07B         | 0.07B        | `siglino-70M`         | [Download](https://github.com/tiiuae/amoe/releases/download/AMoE-checkpoint/dense-70M.pt)        |
| SigLino-30M     | Dense          | 0.03B         | 0.03B        | `siglino-30M`         | [Download](https://github.com/tiiuae/amoe/releases/download/AMoE-checkpoint/dense-30M.pt)        |


## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

## Quick Start

### SigLino-MoE-0.15B (MoE, top-2/28, 0.15B active / 0.6B total)

```python
from siglino import load_siglino_model
from PIL import Image
import torch

# Download checkpoint
# wget https://github.com/tiiuae/amoe/releases/download/AMoE-checkpoint/amoe-0.15B.pt

model, processor = load_siglino_model(
    checkpoint_path="amoe-0.15B.pt",
    config_name="siglino-moe-0.15B",
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
    siglino_features = outputs["patch_features"]["siglino"]  # (N, L, 768)
```

### SigLino-0.6B

```python
from siglino import load_siglino_model
from PIL import Image
import torch

# Download checkpoint
# wget https://github.com/tiiuae/amoe/releases/download/AMoE-checkpoint/dense-0.6B.pt

model, processor = load_siglino_model(
    checkpoint_path="dense-0.6B.pt",
    config_name="siglino-0.6B",
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
    siglino_features = outputs["patch_features"]["siglino"]  # (N, L, 1280)
```

## Results

All results use ensemble features (combined DINOv3 + SigLIP2 heads) unless noted.

### kNN Classification (512x512)


| Model           | Params (active) | ImageNet | CUB-200 | Food-101 | DTD  | Aircraft | Flowers-102 | Avg  |
| --------------- | --------------- | -------- | ------- | -------- | ---- | -------- | ----------- | ---- |
| SigLino-MoE-0.3B | 0.3B            | 85.9     | 88.9    | 95.5     | 80.6 | 92.3     | 99.8        | 90.5 |
| SigLino-MoE-0.15B | 0.15B           | 85.0     | 88.2    | 94.6     | 79.9 | 91.5     | 99.8        | 89.8 |
| SigLino-0.6B    | 0.6B            | 86.1     | 89.2    | 95.8     | 81.1 | 92.3     | 99.8        | 90.7 |
| SigLino-70M     | 0.07B           | 81.7     | 84.8    | 90.2     | 77.3 | 83.3     | 99.6        | 86.2 |
| SigLino-30M     | 0.03B           | 79.0     | 80.9    | 87.5     | 75.7 | 77.3     | 99.2        | 83.3 |


### Zero-Shot Image-Text Classification (512x512)


| Model           | ImageNet | CUB-200 | Food-101 | DTD  | Aircraft | Flowers-102 | Caltech-101 | Avg  |
| --------------- | -------- | ------- | -------- | ---- | -------- | ----------- | ----------- | ---- |
| SigLino-MoE-0.3B | 79.9     | 82.6    | 94.6     | 69.9 | 83.5     | 88.6        | 88.4        | 83.9 |
| SigLino-MoE-0.15B | 78.8     | 80.9    | 93.4     | 69.5 | 81.4     | 89.0        | 88.6        | 83.1 |
| SigLino-0.6B    | 80.5     | 83.0    | 95.0     | 71.1 | 82.4     | 89.0        | 89.9        | 84.4 |
| SigLino-70M     | 71.2     | 70.7    | 85.2     | 65.7 | 60.1     | 84.0        | 88.3        | 75.0 |
| SigLino-30M     | 65.1     | 59.9    | 80.3     | 62.9 | 48.3     | 77.4        | 87.2        | 68.7 |


### Retrieval R@1 (512x512)


| Model           | Flickr30K T2I | Flickr30K I2T | MSCOCO T2I | MSCOCO I2T |
| --------------- | ------------- | ------------- | ---------- | ---------- |
| SigLino-MoE-0.3B | 81.6          | 94.6          | 54.7       | 70.8       |
| SigLino-MoE-0.15B | 81.0          | 92.9          | 54.2       | 71.1       |
| SigLino-0.6B    | 81.9          | 94.2          | 55.6       | 72.9       |
| SigLino-70M     | 77.5          | 90.5          | 50.4       | 65.4       |
| SigLino-30M     | 72.9          | 82.2          | 46.6       | 59.7       |


### Linear Segmentation mIoU (1024x1024)


| Model           | Pascal VOC | Cityscapes |
| --------------- | ---------- | ---------- |
| SigLino-MoE-0.3B | 88.9       | 65.4       |
| SigLino-MoE-0.15B | 88.1       | 63.6       |
| SigLino-0.6B    | 89.8       | 67.3       |
| SigLino-70M     | 84.8       | 61.6       |
| SigLino-30M     | 82.1       | 59.2       |


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

<p align="center">
  <img src="pca_maps_siglino/pca_visualizations/pca_instance.png" width="800">
</p>

## HF usage

```python
import torch
from PIL import Image
from transformers import AutoModel, AutoImageProcessor

# Load model and processor
model_id = "tiiuae/siglino-moe-0.3-0.6B"
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
# Options: 'siglino' (768d), 'siglip2' (1152d), 'dinov3' (1024d)
patch_features = outputs["patch_features"]["siglino"]    # (Batch, Tokens, 768)
summary_features = outputs["summary_features"]["siglip2"] # (Batch, 1152)

```

## Citation

If you use this work in your research, please cite:

```bibtex
@article{chaybouti2025amoe,
  title={AMoE: Agglomerative Mixture-of-Experts Vision Foundation Models},
  author={Chaybouti, Sofian and Narayan, Sanath and Dahou, Yasser and Le Khac, Phuc H. and Singh, Ankit and Huynh, Ngoc Dung and Para, Wamiq Reyaz and Kuehne, Hilde and Hacid, Hakim},
  journal={arXiv preprint arXiv:2512.20157},
  year={2025}
}
```
