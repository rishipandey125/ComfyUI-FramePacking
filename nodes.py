#  Package Modules
import os
from typing import Union, BinaryIO, Dict, List, Tuple, Optional
import time

#  ComfyUI Modules
import folder_paths
from comfy.utils import ProgressBar

import cv2
import numpy as np
import math 
import torch 

from PIL import Image

def convert_tensor_to_numpy(tensor):
    """ Convert tensor to numpy array and scale it properly for image processing. """
    return (tensor.detach().cpu().numpy() * 255).astype(np.uint8)

def convert_numpy_to_tensor(numpy_image):
    """ Convert processed numpy image back to tensor and normalize it. """
    return torch.from_numpy(numpy_image).float() / 255



class ResizeFrame:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frame_width": ("INT", {"default": 4, "min": 1, "tooltip": "Input frame width"}),
                "frame_height": ("INT", {"default": 4, "min": 1, "tooltip": "Input frame height"}),
                "resolution": ("INT", {"default": 768, "min": 1, "tooltip": "Maximum resolution (width * height <= resolution²)"}),
            },
        }

    RETURN_TYPES = ("INT","INT")
    RETURN_NAMES = ("width","height")
    FUNCTION = "resize_frame"
    CATEGORY = "Image Processing"

    def resize_frame(self, frame_width, frame_height, resolution):
        # Ensure resolution is a multiple of 16
        resolution = (resolution // 16) * 16
        max_pixels = resolution * resolution
        
        # Calculate aspect ratio
        aspect_ratio = frame_width / frame_height
        
        # Start with the larger dimension
        if frame_width >= frame_height:
            # Try to maximize width first
            new_width = min(frame_width, resolution)
            new_width = (new_width // 16) * 16
            new_height = int(new_width / aspect_ratio)
            new_height = (new_height // 16) * 16
            
            # If this exceeds max_pixels, scale down proportionally
            if new_width * new_height > max_pixels:
                scale = math.sqrt(max_pixels / (new_width * new_height))
                new_width = int(new_width * scale)
                new_width = (new_width // 16) * 16
                new_height = int(new_width / aspect_ratio)
                new_height = (new_height // 16) * 16
        else:
            # Try to maximize height first
            new_height = min(frame_height, resolution)
            new_height = (new_height // 16) * 16
            new_width = int(new_height * aspect_ratio)
            new_width = (new_width // 16) * 16
            
            # If this exceeds max_pixels, scale down proportionally
            if new_width * new_height > max_pixels:
                scale = math.sqrt(max_pixels / (new_width * new_height))
                new_height = int(new_height * scale)
                new_height = (new_height // 16) * 16
                new_width = int(new_height * aspect_ratio)
                new_width = (new_width // 16) * 16
        
        # Ensure we don't end up with zero dimensions
        new_width = max(16, new_width)
        new_height = max(16, new_height)
        
        return (new_width, new_height)


class ThresholdImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "Input edge maps to binarize"}),
                "threshold": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Threshold from 0.0 to 1.0 — pixels below are black, above are white"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("binarized_images",)
    FUNCTION = "threshold_images"
    CATEGORY = "Image Processing"

    def threshold_images(self, images, threshold):
        images_np = convert_tensor_to_numpy(images)  # shape: [B, H, W, 3], dtype: uint8 or float32

        binarized_images = []
        threshold_255 = threshold * 255.0

        for image in images_np:
            # If image is in float [0, 1], scale it to [0, 255]
            if image.dtype == np.float32 and image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)

            # Grayscale using luminosity method
            grayscale = (0.2989 * image[..., 0] + 0.5870 * image[..., 1] + 0.1140 * image[..., 2])

            # Apply threshold
            binary = (grayscale >= threshold_255).astype(np.uint8) * 255

            # Convert back to RGB
            binarized_rgb = np.stack([binary]*3, axis=-1).astype(np.uint8)
            binarized_images.append(binarized_rgb)

        binarized_np = np.stack(binarized_images, axis=0)
        binarized_tensor = convert_numpy_to_tensor(binarized_np)

        return (binarized_tensor,)




class PadBatchTo4nPlus1:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "Input images to be padded to 4n+1 size"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("padded_images", "original_count", "total_count")
    FUNCTION = "pad_batch"
    CATEGORY = "Image Processing"

    def pad_batch(self, images):
        # Get the current batch size
        current_size = images.shape[0]
        
        # Calculate the next 4n+1 size
        n = math.ceil((current_size - 1) / 4)
        target_size = 4 * n + 1
        
        # If we're already at a 4n+1 size, return as is
        if current_size == target_size:
            return (images, current_size, target_size)
            
        # Calculate how many frames we need to pad
        padding_size = target_size - current_size
        
        # Get the last frame to repeat
        last_frame = images[-1]
        
        # Create the padding frames by repeating the last frame
        padding_frames = last_frame.unsqueeze(0).repeat(padding_size, 1, 1, 1)
        
        # Concatenate the original frames with the padding frames
        padded_images = torch.cat([images, padding_frames], dim=0)
        
        return (padded_images, current_size, target_size)

class TrimPaddedBatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "Padded images to be trimmed"}),
                "original_count": ("INT", {"default": 1, "min": 1, "tooltip": "Original number of frames before padding"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("trimmed_images",)
    FUNCTION = "trim_batch"
    CATEGORY = "Image Processing"

    def trim_batch(self, images, original_count):
        # Ensure original_count is not larger than the current batch size
        original_count = min(original_count, images.shape[0])
        
        # Trim the batch to the original size
        trimmed_images = images[:original_count]
        
        return (trimmed_images,)
