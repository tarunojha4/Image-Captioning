"""Dataset Loaders for Image Captioning.

Provides PyTorch Dataset and Collate classes, as well as a TensorFlow tf.data
pipeline. Integrates OpenCV for high-speed image decoding and augmentation.
"""

import json
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

import config
from src.opencv_utils import augment_image_cv, load_image_cv, normalize_image
from src.vocab import Vocabulary


def load_captions_file(
    captions_file: Union[str, Path]
) -> List[Tuple[str, str]]:
    """Load image-caption pairs from JSON or standard Flickr format text file.

    Returns:
        List of (image_path, caption_str) pairs.
    """
    path = Path(captions_file)
    pairs: List[Tuple[str, str]] = []

    if not path.exists():
        raise FileNotFoundError(f"Annotations file not found: {path}")

    if path.suffix.lower() == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            img_path = item["image_path"]
            for caption in item["captions"]:
                pairs.append((img_path, caption))
    else:
        # Standard Flickr8k style: image,caption per line
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for line in lines:
            parts = line.strip().split(",")
            if len(parts) >= 2:
                img_path = parts[0].strip()
                caption = ",".join(parts[1:]).strip()
                pairs.append((img_path, caption))

    return pairs


class PyTorchCaptionDataset(Dataset):
    """PyTorch Dataset that loads images via OpenCV and numericalizes captions."""

    def __init__(
        self,
        samples: List[Tuple[str, str]],
        vocab: Vocabulary,
        is_train: bool = True,
        augment: bool = True,
    ):
        self.samples = samples
        self.vocab = vocab
        self.is_train = is_train
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, caption = self.samples[idx]

        # 1. Load image using OpenCV
        img_rgb = load_image_cv(img_path, target_size=config.IMAGE_SIZE, to_rgb=True)

        # 2. Augment if training
        if self.is_train and self.augment:
            img_rgb = augment_image_cv(img_rgb)

        # 3. Normalize with ImageNet statistics
        img_norm = normalize_image(img_rgb)

        # 4. Transpose from (H, W, C) to PyTorch tensor format (C, H, W)
        img_tensor = torch.from_numpy(img_norm).permute(2, 0, 1).float()

        # 5. Numericalize caption
        tokens = self.vocab.numericalize(caption, add_special_tokens=True)
        caption_tensor = torch.tensor(tokens, dtype=torch.long)

        return img_tensor, caption_tensor


class CaptionCollate:
    """Custom collate function for PyTorch DataLoader to handle variable length captions."""

    def __init__(self, pad_idx: int = config.PAD_IDX):
        self.pad_idx = pad_idx

    def __call__(
        self, batch: List[Tuple[torch.Tensor, torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Sort batch by caption length in descending order (recommended for RNN/LSTM packing)
        batch.sort(key=lambda x: len(x[1]), reverse=True)

        images = [item[0] for item in batch]
        captions = [item[1] for item in batch]

        # Stack images: (Batch, 3, H, W)
        images = torch.stack(images, dim=0)

        # Lengths of sequences
        lengths = torch.tensor([len(c) for c in captions], dtype=torch.long)

        # Pad captions to maximum length in batch
        padded_captions = torch.nn.utils.rnn.pad_sequence(
            captions, batch_first=True, padding_value=self.pad_idx
        )

        return images, padded_captions, lengths


# =====================================================================
# TensorFlow Data Pipeline
# =====================================================================

def create_tf_dataset(
    samples: List[Tuple[str, str]],
    vocab: Vocabulary,
    batch_size: int = config.BATCH_SIZE,
    is_train: bool = True,
    max_len: int = config.MAX_CAPTION_LEN,
):
    """Create a high-performance tf.data.Dataset pipeline.

    Decodes images via OpenCV and batches padded caption sequences.
    """
    import tensorflow as tf

    def _load_and_process(img_path_bytes, caption_str_bytes):
        img_path = img_path_bytes.numpy().decode("utf-8")
        caption_str = caption_str_bytes.numpy().decode("utf-8")

        # OpenCV load
        img_rgb = load_image_cv(img_path, target_size=config.IMAGE_SIZE, to_rgb=True)
        if is_train:
            img_rgb = augment_image_cv(img_rgb)
        img_norm = normalize_image(img_rgb)  # shape (H, W, 3)

        # Numericalize
        tokens = vocab.numericalize(caption_str, add_special_tokens=True)
        if len(tokens) > max_len:
            tokens = tokens[:max_len]
        else:
            tokens = tokens + [config.PAD_IDX] * (max_len - len(tokens))

        return img_norm.astype(np.float32), np.array(tokens, dtype=np.int32)

    def _tf_map(img_path, caption_str):
        img, cap = tf.py_function(
            func=_load_and_process,
            inp=[img_path, caption_str],
            Tout=[tf.float32, tf.int32],
        )
        img.set_shape([config.IMAGE_SIZE[0], config.IMAGE_SIZE[1], 3])
        cap.set_shape([max_len])
        return img, cap

    img_paths = [s[0] for s in samples]
    caption_texts = [s[1] for s in samples]

    ds = tf.data.Dataset.from_tensor_slices((img_paths, caption_texts))
    if is_train:
        ds = ds.shuffle(buffer_size=max(len(samples), 100))

    ds = ds.map(_tf_map, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds
