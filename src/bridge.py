"""Cross-Framework Bridge & Benchmarking Module.

Connects OpenCV, PyTorch, and TensorFlow to demonstrate deep learning
interoperability, inference benchmarking, and side-by-side comparison rendering.
"""

import time
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import cv2
import numpy as np
import torch

import config
from src.opencv_utils import load_image_cv, normalize_image, overlay_caption_on_image
from src.vocab import Vocabulary


def compare_frameworks(
    image_path: Union[str, Path],
    pytorch_model,
    tf_model,
    vocab: Vocabulary,
    device: torch.device,
    output_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Union[str, float]]:
    """Run an image through both PyTorch and TensorFlow models, benchmarking speed and output.

    Produces a comparative OpenCV visualization banner.
    """
    # 1. OpenCV Preprocessing
    img_rgb = load_image_cv(image_path, target_size=config.IMAGE_SIZE, to_rgb=True)
    img_norm = normalize_image(img_rgb)

    # 2. PyTorch Inference
    t0 = time.perf_counter()
    img_pt = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).float().to(device)
    caption_pt, alphas_pt, score_pt = pytorch_model.generate(
        img_pt, vocab, method="beam", beam_width=3
    )
    t_pt = (time.perf_counter() - t0) * 1000.0

    # 3. TensorFlow Inference
    t0 = time.perf_counter()
    caption_tf, alphas_tf = tf_model.predict_caption(img_norm, vocab)
    t_tf = (time.perf_counter() - t0) * 1000.0

    # 4. Generate Side-by-Side OpenCV Comparison Visualization
    # Read original image in BGR
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    h, w = img_bgr.shape[:2]
    vis_h = max(380, h)
    vis_w = max(500, w)
    resized_bgr = cv2.resize(img_bgr, (vis_w, vis_h))

    # Add dual banners: Top banner for PyTorch, Bottom banner for TensorFlow
    banner_h = 75
    overlay = resized_bgr.copy()

    # PyTorch Header (Top)
    cv2.rectangle(overlay, (0, 0), (vis_w, banner_h), (20, 20, 20), -1)
    # TensorFlow Footer (Bottom)
    cv2.rectangle(overlay, (0, vis_h - banner_h), (vis_w, vis_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.82, resized_bgr, 0.18, 0, resized_bgr)

    # Decorative separator lines
    cv2.line(resized_bgr, (0, banner_h), (vis_w, banner_h), (0, 215, 255), 2)
    cv2.line(resized_bgr, (0, vis_h - banner_h), (vis_w, vis_h - banner_h), (50, 205, 50), 2)

    # PyTorch Text
    cv2.putText(
        resized_bgr,
        f"[PyTorch ResNet50 + BeamSearch] ({t_pt:.1f} ms)",
        (12, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (0, 215, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        resized_bgr,
        f'"{caption_pt}"',
        (12, 54),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # TensorFlow Text
    cv2.putText(
        resized_bgr,
        f"[TensorFlow MobileNetV2 + GRU] ({t_tf:.1f} ms)",
        (12, vis_h - banner_h + 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (50, 205, 50),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        resized_bgr,
        f'"{caption_tf}"',
        (12, vis_h - banner_h + 54),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), resized_bgr)

    return {
        "pytorch_caption": caption_pt,
        "pytorch_latency_ms": round(t_pt, 2),
        "tensorflow_caption": caption_tf,
        "tensorflow_latency_ms": round(t_tf, 2),
    }
