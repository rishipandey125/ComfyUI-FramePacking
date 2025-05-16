#  Package Modules
import os
from typing import Union, BinaryIO, Dict, List, Tuple, Optional
import time

#  ComfyUI Modules
import folder_paths
from comfy.utils import ProgressBar, common_upscale

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
        # Calculate maximum allowed pixels
        max_pixels = resolution * resolution
        
        # Calculate aspect ratio
        aspect_ratio = frame_width / frame_height
        
        # Calculate initial dimensions based on aspect ratio
        if frame_width >= frame_height:
            # Start with width
            new_width = frame_width
            new_height = int(new_width / aspect_ratio)
        else:
            # Start with height
            new_height = frame_height
            new_width = int(new_height * aspect_ratio)
        
        # If the initial dimensions exceed max_pixels, scale down proportionally
        if new_width * new_height > max_pixels:
            scale = math.sqrt(max_pixels / (new_width * new_height))
            new_width = int(new_width * scale)
            new_height = int(new_height * scale)
        
        # Ensure dimensions are multiples of 16
        new_width = (new_width // 16) * 16
        new_height = (new_height // 16) * 16
        
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

class GetImageDimensions:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Input image to get dimensions from"}),
            },
        }

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("width", "height")
    FUNCTION = "get_dimensions"
    CATEGORY = "Image Processing"

    def get_dimensions(self, image):
        # Get dimensions from the image tensor
        # Image tensor shape is [batch, height, width, channels]
        height, width = image.shape[1:3]
        return (width, height)

class SlotFrame:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "num_frames": ("INT", {"default": 81, "min": 1, "max": 10000, "step": 4, "tooltip": "Number of frames to encode"}),
            "empty_frame_level": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "White level of empty frame to use"}),
            "slot_image": ("IMAGE", {"tooltip": "Image to place in the specified slot"}),
            "slot_index": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1, "tooltip": "Index where to place the slot image"}),
            },
            "optional": {
                "control_images": ("IMAGE",),
                "inpaint_mask": ("MASK", {"tooltip": "Inpaint mask to use for the empty frames"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", )
    RETURN_NAMES = ("images", "masks",)
    FUNCTION = "process"
    CATEGORY = "Image Processing"
    DESCRIPTION = "Places a single image at a specified index in a sequence of frames"

    def process(self, num_frames, empty_frame_level, slot_image, slot_index, control_images=None, inpaint_mask=None):
        B, H, W, C = slot_image.shape
        device = slot_image.device

        masks = torch.ones((num_frames, H, W), device=device)

        if control_images is not None:
            control_images = common_upscale(control_images.movedim(-1, 1), W, H, "lanczos", "disabled").movedim(1, -1)
        
        # Create empty frames
        if control_images is None:
            empty_frames = torch.ones((num_frames, H, W, 3), device=device) * empty_frame_level
        else:
            empty_frames = control_images[:num_frames]
        
        # Ensure slot_index is within bounds
        slot_index = min(slot_index, num_frames - 1)
        
        # Replace the frame at slot_index with slot_image
        empty_frames[slot_index:slot_index + slot_image.shape[0]] = slot_image
        
        # Create mask for the slot image
        masks[slot_index:slot_index + slot_image.shape[0]] = 0

        if inpaint_mask is not None:
            inpaint_mask = common_upscale(inpaint_mask.unsqueeze(1), W, H, "nearest-exact", "disabled").squeeze(1).to(device)
            if inpaint_mask.shape[0] > num_frames:
                inpaint_mask = inpaint_mask[:num_frames]
            elif inpaint_mask.shape[0] < num_frames:
                inpaint_mask = inpaint_mask.repeat(num_frames // inpaint_mask.shape[0] + 1, 1, 1)[:num_frames]

            empty_mask = torch.ones_like(masks, device=device)
            masks = inpaint_mask * empty_mask
    
        return (empty_frames.cpu().float(), masks.cpu().float())
