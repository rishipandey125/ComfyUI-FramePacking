from .nodes import *


#  Map all your custom nodes classes with the names that will be displayed in the UI.

#nodes I will need 

#Pack Frames 
#Unpack Frames

NODE_CLASS_MAPPINGS = {
    "Add Grid Boundaries": AddGridBoundaries,
    "Pack Frames": PackFrames,
    "Unpack Frames": UnpackFrames,
    "Resize Frame": ResizeFrame,
}


__all__ = ['NODE_CLASS_MAPPINGS']
