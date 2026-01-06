
import random
import numpy as np
import torch
import json
import os
from typing import Dict, List
from PIL import Image
from .schemas import Example


def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        config = json.load(file)
    return config


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_dataset(dataset_config_path: str) -> List[Example]:
    raw: Dict = load_json(dataset_config_path)
    examples: List[Example] = []
    for key, d in raw.items():
        examples.append(Example.from_dict(key, d))
    return examples


def load_image(base_image_path: str, rel_path: str) -> Image.Image:
    path = os.path.join(base_image_path, rel_path)
    return Image.open(path).convert("RGB")


def resize_to_fit(img: Image.Image, height: int, width: int) -> Image.Image:
    ratio = min(height / img.height, width / img.width)
    return img.resize((int(img.width * ratio), int(img.height * ratio)))


def save_first_image(images, save_path: str, height: int, width: int) -> None:
    img = images[0]
    img = resize_to_fit(img, height, width)
    img.save(save_path)
