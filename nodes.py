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
    def INPUT_TYPES(cls):
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

    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("sheets", "frame_width", "frame_height")
    FUNCTION = "pack_frames"
    CATEGORY = "FramePack"

    def pack_frames(self, images, num_sheets, sheet_width, sheet_height, min_frame_width, min_frame_height):
        images_np = convert_tensor_to_numpy(images)

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
        return torch.stack(packed_sheets_tensor), frame_width, frame_height

class UnpackFrames:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "Input images to be packed"}),
                "frame_width": ("INT", {"default": 512, "min": 1, "max": 5120, "step": 1}),
                "frame_height": ("INT", {"default": 512, "min": 1, "max": 5120, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("frames",)
    FUNCTION = "unpack_frames"
    CATEGORY = "FramePack"

    def unpack_frames(self, images, frame_width, frame_height):
        # Convert tensor of images to numpy for processing
        images_np = convert_tensor_to_numpy(images)

        unpacked_frames = []
        
        for image in images_np:
            rows = image.shape[0] // frame_height
            cols = image.shape[1] // frame_width
            for row in range(rows):
                for col in range(cols):
                    x = col * frame_width
                    y = row * frame_height
                    frame = image[y:y + frame_height, x:x + frame_width]
                    # Check if the frame is completely black
                    if np.any(frame > 0):  # Skip completely black frames
                        unpacked_frames.append(frame)
        
        # Convert unpacked frames back to tensors
        unpacked_frames_tensors = [convert_numpy_to_tensor(frame) for frame in unpacked_frames]
        return torch.stack(unpacked_frames_tensors),