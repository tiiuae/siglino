# Image preprocessing for Falcon Vision
# Handles resizing, normalization, and patchification

import math
import numpy as np
import torch
from PIL import Image


IMAGE_MEAN = [0.5, 0.5, 0.5]
IMAGE_STD = [0.5, 0.5, 0.5]


def smart_resize(
    height: int,
    width: int,
    factor: int = 16,
    min_pixels: int = 128 * 128,
    max_pixels: int = 256 * 256,
) -> tuple[int, int]:
    """Resize dimensions to be divisible by factor while respecting pixel bounds."""
    if height < factor or width < factor:
        raise ValueError(f"height:{height} or width:{width} must be larger than factor:{factor}")
    if max(height, width) / min(height, width) > 200:
        raise ValueError(f"absolute aspect ratio must be smaller than 200")
    
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    
    if h_bar * w_bar > max_pixels:
        beta = np.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = np.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    
    return h_bar, w_bar


def convert_image_to_patches(image: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Convert image (H, W, C) to patches (num_patches, patch_size^2 * C)."""
    image_height, image_width, num_channels = image.shape
    num_patches_height = image_height // patch_size
    num_patches_width = image_width // patch_size
    
    patched_image = image.reshape(num_patches_height, patch_size, num_patches_width, patch_size, num_channels)
    patched_image = patched_image.permute(0, 2, 1, 3, 4)
    patched_image = patched_image.reshape(num_patches_height * num_patches_width, -1)
    return patched_image


def pad_along_first_dim(
    array: torch.Tensor,
    target_length: int,
    pad_value: float = 0.0,
    mask_dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad the array along the first dimension and return mask."""
    current_length = array.shape[0]
    padding_length = target_length - current_length
    mask = torch.ones(target_length, dtype=mask_dtype, device=array.device)
    
    if padding_length > 0:
        paddings = (0, 0, 0, padding_length)
        array = torch.nn.functional.pad(array, paddings, mode="constant", value=pad_value)
        mask[-padding_length:] = 0
    
    return array, mask


class AMOEImageProcessor:
    """Image processor for AMOE model.
        """
    
    def __init__(
        self,
        patch_size: int = 16,
        min_pixels: int = 128 * 128,
        max_pixels: int = 256 * 256,
        image_mean: list[float] | None = None,
        image_std: list[float] | None = None,
        do_resize: bool = True,
        do_rescale: bool = True,
        do_normalize: bool = True,
    ):
        self.patch_size = patch_size
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.image_mean = image_mean or IMAGE_MEAN
        self.image_std = image_std or IMAGE_STD
        self.do_resize = do_resize
        self.do_rescale = do_rescale
        self.do_normalize = do_normalize
    
    def preprocess_single(self, image: Image.Image | np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
        """Preprocess a single image."""
        if isinstance(image, Image.Image):
            image = image.convert("RGB")
            image = np.array(image)
        
        # Ensure HWC format
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[0] == 3:  # CHW -> HWC
            image = np.transpose(image, (1, 2, 0))
        
        height, width = image.shape[:2]
        
        # Smart resize
        if self.do_resize:
            resized_height, resized_width = smart_resize(
                height, width,
                factor=self.patch_size,
                min_pixels=self.min_pixels,
                max_pixels=self.max_pixels,
            )
            pil_image = Image.fromarray(image.astype(np.uint8))
            pil_image = pil_image.resize((resized_width, resized_height), Image.BICUBIC)
            image = np.array(pil_image)
        else:
            resized_height, resized_width = height, width
        
        # Rescale to [0, 1]
        if self.do_rescale:
            image = image.astype(np.float32) / 255.0
        
        # Normalize
        if self.do_normalize:
            mean = np.array(self.image_mean, dtype=np.float32)
            std = np.array(self.image_std, dtype=np.float32)
            image = (image - mean) / std
        
        spatial_shape = (resized_height // self.patch_size, resized_width // self.patch_size)
        return image, spatial_shape
    
    def preprocess(
        self,
        images: list[Image.Image] | list[np.ndarray],
    ) -> tuple[list[np.ndarray], list[tuple[int, int]]]:
        """Preprocess a list of images."""
        pixel_values = []
        spatial_shapes = []
        
        for image in images:
            processed_image, spatial_shape = self.preprocess_single(image)
            pixel_values.append(processed_image)
            spatial_shapes.append(spatial_shape)
        
        return pixel_values, spatial_shapes
    
    def batch_images_with_mask(
        self,
        pixel_values: list[np.ndarray],
        spatial_shapes: list[tuple[int, int]],
        max_num_patches: int = 256,
        pad: bool = True,
        output_dtype: torch.dtype = torch.float32,
        mask_dtype: torch.dtype | None = None,
    ) -> dict[str, torch.Tensor]:
        """Batch images into padded tensors with masks.
        """
        if not pixel_values:
            return None

        if mask_dtype is None:
            mask_dtype = output_dtype
        
        batched_pixels = []
        batched_masks = []
        batched_shapes = []
        
        for img, shape in zip(pixel_values, spatial_shapes):
            img_tensor = torch.from_numpy(img).to(dtype=output_dtype)
            patches = convert_image_to_patches(img_tensor, self.patch_size)
            
            if pad:
                patches, mask = pad_along_first_dim(
                    patches,
                    max_num_patches,
                    mask_dtype=mask_dtype,
                )
            else:
                mask = torch.ones(patches.shape[0], dtype=mask_dtype, device=patches.device)
            
            batched_pixels.append(patches)
            batched_masks.append(mask)
            batched_shapes.append(list(shape))
        
        return {
            "pixel_values": torch.stack(batched_pixels),
            "padding_mask": torch.stack(batched_masks),
            "spatial_shape": torch.tensor(batched_shapes),
        }
    
    def __call__(
        self,
        images: list[Image.Image] | Image.Image,
        max_num_patches: int = 256,
        n_storage_tokens: int = 4,  # kept for API compat, not used here
        return_tensors: str = "pt",
        pad: bool = True,
        output_dtype: torch.dtype = torch.float32,
        mask_dtype: torch.dtype | None = None,
    ) -> dict[str, torch.Tensor]:
        """Process images and return batched tensors."""
        if isinstance(images, Image.Image):
            images = [images]
        
        pixel_values, spatial_shapes = self.preprocess(images)
        
        return self.batch_images_with_mask(
            pixel_values,
            spatial_shapes,
            max_num_patches=max_num_patches,
            pad=pad,
            output_dtype=output_dtype,
            mask_dtype=mask_dtype,
        )
