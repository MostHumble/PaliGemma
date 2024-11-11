import numpy as np
from PIL import Image
import torch

IMAGENET_MN = [0.5, 0.5, 0.5]
IMAGENET_STD = [0.5, 0.5, 0.5]

class PaliGemmaProcessor:
    def __init__(self, tokenizer, num_image_tokens: int, image_size:int):

        