from typing import Any, List, Union, Optional
import numpy as np
from PIL import Image
import torch

IMAGENET_MN = [0.5, 0.5, 0.5]
IMAGENET_STD = [0.5, 0.5, 0.5]

def process_images(
        images: List[Image, Image],
        size: int = None,
        resample: Image.Resampling = None,
        rescale_factor: float = None,
        image_mean : Optional[Union[float, List[float]]] = None,
        image_std : Optional[Union[float, List[float]]] = None,

) -> List[np.ndarry]:
    h, w = size, size
    # resize, rescale, and normalize
    images = [
        resize(image=image, size=(h,w), resample=resample) for image in images
    ]
    images = [np.array(image) for image in images]

    images = [rescale(image) for image in images]

    images = [normalize(image) for image in images]

    images = [image.transpose(2, 0, 1) for image in images]

    return images

class PaliGemmaProcessor:
    def __init__(self, tokenizer, num_image_tokens: int, image_size:int):

        self.image_seq_lenght = num_image_tokens
        self.image_size = image_size

        tokens_to_add = {"additional_special_tokens": {self.IMAGE_TOKEN}}
        tokenizer.add_special_tokens(tokens_to_add)

        EXTRA_TOKENS = [
            f"<loc{i:04d}>" for i in range(1024)
            ]
        EXTRA_TOKENS += [
            f"<seg{i:03d}" for i in range(128)
        ]

        tokenizer.add_special_tokens(EXTRA_TOKENS)

        self.image_token_id = tokenizer.convert_tokens_to_ids(self.IMAGE_TOKEN)

        tokenizer.add_bos_token = False
        tokenizer.add_eos_token = False

        self.tokenizer = tokenizer
        
    def __call__(self,
                 text: List[str],
                 images: List[Image, Image],
                 padding: str = "longest",
                 truncation : bool = True,
    ) -> dict :
        
        assert len(images) == 1 and len(text) == 1, f'only accepts 1 image and 1 text at a time'

        pixel_values = process_images(
            images,
            size=(self.image_size, self.image_size),
            rescale_factor = 1/255.0,
            image_mean = IMAGENET_MN,
            image_std = IMAGENET_STD 
        )
        # bs, c, h, w
        pixel_values = np.stack(pixel_values, axis=0)
        pixel_values = torch.tensor(pixel_values)

        input_strings = [
            add_image_tokens_to_prompt(
                prefix_prompt=prompt,
                bos_token=self.tokenizer.bos_token,
                image_seq_len=self.image_seq_lenght,
                image_tokens=self.IMAGE_TOKEN) 
                for prompt in text
            ]

        inputs = self.tokenizer(
            input_strings,
            return_tensors="pt",
            padding=padding,
            truncation=truncation
        )

        return_data = {"pixel_values": pixel_values, **inputs}

        return return_data
        