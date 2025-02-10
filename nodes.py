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
                "resolution": ("INT", {"default": 768, "min": 1, "tooltip": "Resize resolution"}),
            },
        }

    RETURN_TYPES = ("INT","INT")
    RETURN_NAMES = ("width","height")
    FUNCTION = "resize_frame"
    CATEGORY = "Image Processing"

    def resize_frame(self, frame_width, frame_height, resolution):
        width = frame_width
        height = frame_height
        
        if (frame_width > resolution or frame_height > resolution): 
            if frame_width > frame_height: 
                width = resolution
                height = resolution * (frame_height / frame_width)
            else:
                height = resolution
                width = resolution * (frame_width / frame_height)
        width = int(width)
        height = int(height)
        
        return (width, height)

class AddGridBoundaries:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "Input images (edge maps) to add grid boundaries"}),
                "cells_per_row": ("INT", {"default": 4, "min": 1, "tooltip": "Number of grid cells per row"}),
                "cells_per_col": ("INT", {"default": 4, "min": 1, "tooltip": "Number of grid cells per column"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images_with_grid",)
    FUNCTION = "add_grid_boundaries"
    CATEGORY = "Image Processing"

    def add_grid_boundaries(self, images, cells_per_row, cells_per_col):
        # Convert tensor images to numpy array
        images_np = convert_tensor_to_numpy(images)
        
        # Iterate over each image to add grid lines
        for image in images_np:
            height, width, channels = image.shape
            row_height = height // cells_per_col
            col_width = width // cells_per_row
            
            # Draw grid lines
            for i in range(1, cells_per_row):
                x = i * col_width
                image[:, x - 1:x + 1] = 255  # Vertical grid line, adjust thickness as needed
            
            for j in range(1, cells_per_col):
                y = j * row_height
                image[y - 1:y + 1, :] = 255  # Horizontal grid line, adjust thickness as needed
        
        # Convert numpy images back to tensor
        images_with_grid_tensor = convert_numpy_to_tensor(images_np)
        
        return (images_with_grid_tensor,)

class PackFrames:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "Input images to be packed"}),
            },
            "optional": {
                "num_sheets": ("INT", {"default": 8, "min": 1, "max": 64, "step": 1}),
                "sheet_width": ("INT", {"default": 2048, "min": 1024, "max": 5120, "step": 512}),
                "sheet_height": ("INT", {"default": 2048, "min": 1024, "max": 5120, "step": 512}),
                "min_frame_width": ("INT", {"default": 512, "min": 512, "max": 5120, "step": 512}),
                "min_frame_height": ("INT", {"default": 512, "min": 512, "max": 5120, "step": 512}),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("sheets", "frame_width", "frame_height", "cells_per_row", "cells_per_col")
    FUNCTION = "pack_frames"
    CATEGORY = "FramePack"

    def pack_frames(self, images, num_sheets, sheet_width, sheet_height, min_frame_width, min_frame_height):
        images_np = convert_tensor_to_numpy(images)

        total_frames = images_np.shape[0]
        frames_per_sheet = math.ceil(total_frames / num_sheets)
        side_count = math.ceil(math.sqrt(frames_per_sheet))
        frame_width = max(min_frame_width, sheet_width // side_count)
        frame_height = max(min_frame_height, sheet_height // side_count)
        cells_per_row = sheet_width // frame_width
        cells_per_col = sheet_height // frame_height

        packed_sheets = []
        current_frame = 0

        for sheet_index in range(num_sheets):
            sprite_sheet = np.zeros((sheet_height, sheet_width, 3), dtype=np.uint8)
            for row in range(cells_per_col):
                for col in range(cells_per_row):
                    if current_frame < total_frames:
                        frame = images_np[current_frame]
                        resized_frame = cv2.resize(frame, (frame_width, frame_height), interpolation=cv2.INTER_AREA)
                    else:
                        resized_frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)  # Black frame

                    x = col * frame_width
                    y = row * frame_height
                    sprite_sheet[y:y + frame_height, x:x + frame_width] = resized_frame
                    current_frame += 1
            packed_sheets.append(sprite_sheet)

        packed_sheets_tensor = [convert_numpy_to_tensor(sheet) for sheet in packed_sheets]
        return torch.stack(packed_sheets_tensor), frame_width, frame_height, cells_per_row, cells_per_col

class UnpackFrames:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "Sprite sheets containing packed images"}),
                "frame_width": ("INT", {"default": 512, "min": 1, "max": 5120, "step": 1}),
                "frame_height": ("INT", {"default": 512, "min": 1, "max": 5120, "step": 1}),
            },
            "optional": {
                "frame_count": ("INT", {"default": None, "min": 1, "tooltip": "Maximum number of frames to extract"})
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("frames",)
    FUNCTION = "unpack_frames"
    CATEGORY = "FrameUnpack"

    def unpack_frames(self, images, frame_width, frame_height, frame_count=None):
        # Convert tensor of images to numpy for processing
        images_np = convert_tensor_to_numpy(images)

        unpacked_frames = []
        current_frame_count = 0
        
        for image in images_np:
            rows = image.shape[0] // frame_height
            cols = image.shape[1] // frame_width
            for row in range(rows):
                for col in range(cols):
                    if frame_count is not None and current_frame_count >= frame_count:
                        break
                    x = col * frame_width
                    y = row * frame_height
                    frame = image[y:y + frame_height, x:x + frame_width]
                    unpacked_frames.append(frame)
                    current_frame_count += 1
                if frame_count is not None and current_frame_count >= frame_count:
                    break
            if frame_count is not None and current_frame_count >= frame_count:
                break
        
        # Convert unpacked frames back to tensors
        unpacked_frames_tensors = [convert_numpy_to_tensor(frame) for frame in unpacked_frames]
        return torch.stack(unpacked_frames_tensors),
