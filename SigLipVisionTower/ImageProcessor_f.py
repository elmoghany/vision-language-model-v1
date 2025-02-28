import torch
import torch.nn as nn
import math
import numpy as np
from PIL import Image
from typing import Tuple, Optional, Union, Iterable, List, Dict

def add_image_tokens_to_prompt(prefix_prompt, bos_token, image_seq_len, image_token):
    # The function constructs a new prompt string by:
    # 1. Repeating the image token (e.g., "<image>") a number of times (image_seq_len).
    # 2. Adding the beginning-of-sequence token (bos_token).
    # 3. Appending the actual text prompt (prefix_prompt).
    # 4. Ending with a newline character.
    return f"{image_token * image_seq_len}{bos_token}{prefix_prompt}\n"

def rescale(image: np.ndarray, scale: float, dtype: np.dtype = np.float32) -> np.ndarray:
    rescaled_image = image * scale
    rescaled_image = rescaled_image.astype(dtype)
    return rescaled_image

def resize(
    image: Image,
    size: Tuple[int, int],
    resample: Image.Resampling = None,
    reducing_gap: Optional[int] = None,
) -> np.ndarray:
    height, width = size
    resized_image = image.resize(
        (width, height), resample=resample, reducing_gap=reducing_gap
    )
    return resized_image

def normalize(
    image: np.ndarray,
    mean: Union[float, Iterable[float]],
    std: Union[float, Iterable[float]],
) -> np.ndarray:
    mean = np.array(mean, dtype=image.dtype)
    std = np.array(std, dtype=image.dtype)
    image = (image - mean) / std
    return image

def process_images(
    images: List[Image.Image],
    size: Dict[str, int] = None,
    resample: Image.Resampling = None,
    rescale_factor: float = None,
    image_mean: Optional[Union[float, List[float]]] = None,
    image_std: Optional[Union[float, List[float]]] = None,
) -> List[np.ndarray]:
    height, width = size[0], size[1]
    images = [
        resize(image=image, size=(height, width), resample=resample) for image in images
    ]
    # Convert each image to a NumPy array
    images = [np.array(image) for image in images]
    # Rescale the pixel values to be in the range [0, 1]
    images = [rescale(image, scale=rescale_factor) for image in images]
    # Normalize the images to have mean 0 and standard deviation 1
    images = [normalize(image, mean=image_mean, std=image_std) for image in images]
    # Move the channel dimension to the first dimension.
    # The model expects images in the format [Channel, Height, Width]
    images = [image.transpose(2, 0, 1) for image in images]
    return images

IMAGENET_STANDARD_MEAN = [0.5, 0.5, 0.5]
IMAGENET_STANDARD_STD = [0.5, 0.5, 0.5]

class ImageProcessor:
    IMAGE_TOKEN = "<image>"
    
    def __init__(
        self,
        tokenizer,
        num_image_tokens: int,
        image_size: int
    ):
        super().__init__()
        
        self.image_seq_len = num_image_tokens
        self.image_size = image_size
        
        # Add the special image token to the tokenizer.
        tokens_to_add = {"additional_special_tokens": [self.IMAGE_TOKEN]}
        self.tokenizer.add_special_tokens(tokens_to_add)
        
        # extra token for object detection bounding box & segmentation
        EXTRA_TOKENS = [f"<loc{i:04d}>" for i in range(1024)]
        EXTRA_TOKENS += [f"<seg{i:03d}>" for i in range(128)]
        self.tokenizer.add_tokens(EXTRA_TOKENS)
        
        # conver image token to token id
        self.image_token_id = self.tokenizer.convert_tokens_to_ids(self.IMAGE_TOKEN)
        
        # handle BOS beginning of sequence & EOS end of sequence manually
        self.tokenizer.add_bos_token = False
        self.tokenizer.add_eos_token = False
        
    def __call__(
        self,
        text: List[str],
        images: List[Image.Image],
        padding: str = "longest",
        truncation: bool = True
    ) -> dict:
        assert len(images) == 1 and len(text) == 1, f"number of images= {len(images)} and number of prompts= {len(text)}"
        
        pixel_values = process_images(
            images,
            size=(self.image_size, self.image_size),
            resample=Image.Resampling.BICUBIC,
            rescale_factor=1/255.0,
            image_mean=IMAGENET_STANDARD_MEAN,
            image_std=IMAGENET_STANDARD_STD
        )
        
        # add batch size
        # If you have one image with shape [3, 224, 224], 
        # stacking it results in a NumPy array of shape [1, 3, 224, 224].
        # [Batch_Size, Channel, Height, Width]
        pixel_values = np.array(pixel_values, axis=0)
        pixel_values = torch.tensor(pixel_values)

        # Prepend a `self.image_seq_length` number of image tokens to the prompt.
        input_strings = [
            add_image_tokens_to_prompt(
                prefix_prompt = prompt,
                bos_token = self.tokenizer.bos_token, 
                image_seq_len = self.image_seq_len, 
                image_token = self.IMAGE_TOKEN
            )
            for prompt in text
        ]
        
        # Returns the input_ids and attention_mask as PyTorch tensors
        inputs = self.tokenizer(
            input_strings,
            return_tensors = "pt",
            padding = padding,
            truncation = truncation
        )
        
        return_data = {"pixel_values": pixel_values, **inputs}
        
        return return_data
    