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

def convert_tensor_to_cv2_images(tensor):
    images_np = (tensor.numpy() * 255).astype(np.uint8)  # Assuming the tensor is from PyTorch and needs conversion
    images = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in images_np]
    return images

def convert_numpy_to_pil(numpy_image):
    if numpy_image.ndim == 3 and numpy_image.shape[2] == 3:  # Typical HWC format for color images
        return Image.fromarray(cv2.cvtColor(numpy_image, cv2.COLOR_BGR2RGB), 'RGB')
    else:
        raise ValueError("Image array shape is not suitable for conversion to PIL Image")


class PackFrames:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "Input images to be packed"}),
            },
            "optional": {
                "num_sheets": ("INT", {"default": 8, "min": 8, "max": 64, "step": 8}),
                "sheet_width": ("INT", {"default": 2048, "min": 1024, "max": 5120, "step": 512}),
                "sheet_height": ("INT", {"default": 2048, "min": 1024, "max": 5120, "step": 512}),
                "min_frame_width": ("INT", {"default": 512, "min": 512, "max": 5120, "step": 512}),
                "min_frame_height": ("INT", {"default": 512, "min": 512, "max": 5120, "step": 512}),
            },
        }
    OUTPUT_NODE = True
    RETURN_TYPES = ("IMAGE", "INT", "INT",)
    RETURN_NAMES = ("sheets", "frame_width", "frame_height")
    FUNCTION = "pack_frames"
    CATEGORY = "FramePack"

    def pack_frames(self, images, num_sheets, sheet_width, sheet_height, min_frame_width, min_frame_height):
        images_cv2 = convert_tensor_to_cv2_images(images)

        # Calculate frames and dimensions
        total_frames = len(images_cv2)
        frames_per_sheet = math.ceil(total_frames / num_sheets)
        side_count = math.ceil(math.sqrt(frames_per_sheet))
        frame_width = max(min_frame_width, sheet_width // side_count)
        frame_height = max(min_frame_height, sheet_height // side_count)

        # Pack the frames into sprite sheets
        packed_sheets = []
        current_frame = 0
        for sheet_index in range(num_sheets):
            sprite_sheet = np.zeros((sheet_height, sheet_width, 3), dtype=np.uint8)
            for row in range(side_count):
                for col in range(side_count):
                    if current_frame >= total_frames:
                        break
                    frame = images_cv2[current_frame]
                    resized_frame = cv2.resize(frame, (frame_width, frame_height), interpolation=cv2.INTER_AREA)
                    x = col * frame_width
                    y = row * frame_height
                    sprite_sheet[y:y + frame_height, x:x + frame_width] = resized_frame
                    current_frame += 1
                if current_frame >= total_frames:
                    break
            packed_sheets.append(sprite_sheet)
            if current_frame >= total_frames:
                break

        packed_sheets_pil = [convert_numpy_to_pil(sheet) for sheet in packed_sheets if sheet.ndim == 3 and sheet.shape[2] == 3]
        print(packed_sheets_pil)
        return (packed_sheets_pil, frame_width, frame_height,)
