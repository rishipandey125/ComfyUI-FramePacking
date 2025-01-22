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
        images_np = convert_tensor_to_numpy(images)

        # Calculate frames and dimensions
        total_frames = images_np.shape[0]
        frames_per_sheet = math.ceil(total_frames / num_sheets)
        side_count = math.ceil(math.sqrt(frames_per_sheet))
        frame_width = max(min_frame_width, sheet_width // side_count)
        frame_height = max(min_frame_height, sheet_height // side_count)

        packed_sheets = []
        current_frame = 0
        for sheet_index in range(num_sheets):
            sprite_sheet = np.zeros((sheet_height, sheet_width, 3), dtype=np.uint8)
            for row in range(side_count):
                for col in range(side_count):
                    if current_frame >= total_frames:
                        break
                    frame = images_np[current_frame]
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


        # packed_sheets_tensor = numpy_to_tensor(packed_sheets)
        packed_sheets_tensor = [convert_numpy_to_tensor(sheet) for sheet in packed_sheets]


        return (torch.stack(packed_sheets_tensor), frame_width, frame_height,)
