"""OpenCV Computer Vision Utilities for Image Captioning.

Provides image loading, aspect-ratio preserving letterboxing, data augmentation,
attention heatmap visualization, caption HUD overlays, and synthetic dataset generation.
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union
import cv2
import numpy as np

import config


def load_image_cv(
    image_path: Union[str, Path],
    target_size: Tuple[int, int] = config.IMAGE_SIZE,
    to_rgb: bool = True,
    letterbox: bool = False,
) -> np.ndarray:
    """Load an image using OpenCV and resize to target dimension.

    Args:
        image_path: Path to the image file.
        target_size: (height, width) tuple.
        to_rgb: Whether to convert BGR to RGB.
        letterbox: If True, pads borders to preserve original aspect ratio.

    Returns:
        np.ndarray: Image array of shape (H, W, 3) in uint8 [0, 255].
    """
    path_str = str(image_path)
    img_bgr = cv2.imread(path_str)
    if img_bgr is None:
        raise FileNotFoundError(f"Failed to read image at: {path_str}")

    target_h, target_w = target_size

    if letterbox:
        h, w = img_bgr.shape[:2]
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Pad with gray border
        canvas = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        pad_top = (target_h - new_h) // 2
        pad_left = (target_w - new_w) // 2
        canvas[pad_top : pad_top + new_h, pad_left : pad_left + new_w] = resized
        img = canvas
    else:
        img = cv2.resize(img_bgr, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    if to_rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalize uint8 RGB image [0, 255] to float32 using ImageNet mean & std."""
    img_float = image.astype(np.float32) / 255.0
    mean = np.array(config.IMAGENET_MEAN, dtype=np.float32)
    std = np.array(config.IMAGENET_STD, dtype=np.float32)
    normalized = (img_float - mean) / std
    return normalized


def augment_image_cv(image_rgb: np.ndarray) -> np.ndarray:
    """Apply OpenCV computer vision data augmentations.

    Includes random horizontal flip, brightness/contrast jitter,
    and slight affine transformation.
    """
    img = image_rgb.copy()
    h, w = img.shape[:2]

    # 1. Random horizontal flip (50% probability)
    if np.random.rand() > 0.5:
        img = cv2.flip(img, 1)

    # 2. Random brightness and contrast adjustment
    if np.random.rand() > 0.3:
        alpha = np.random.uniform(0.85, 1.15)  # Contrast
        beta = np.random.uniform(-15, 15)     # Brightness
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    # 3. Random slight rotation / scaling
    if np.random.rand() > 0.5:
        angle = np.random.uniform(-10, 10)
        scale = np.random.uniform(0.95, 1.05)
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, scale)
        img = cv2.warpAffine(img, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)

    return img


def overlay_caption_on_image(
    image: np.ndarray,
    caption: str,
    model_tag: str = "PyTorch",
    confidence: Optional[float] = None,
    output_path: Optional[Union[str, Path]] = None,
) -> np.ndarray:
    """Overlay a styled caption banner and HUD information onto an image.

    Args:
        image: RGB or BGR numpy array.
        caption: Caption string to display.
        model_tag: Framework label (e.g. 'PyTorch + Attention', 'TensorFlow GRU').
        confidence: Optional confidence or probability score.
        output_path: If provided, saves the output image to this path.

    Returns:
        np.ndarray: BGR image suitable for cv2.imshow or cv2.imwrite.
    """
    # Ensure working in BGR for OpenCV rendering
    if image.dtype != np.uint8:
        display_img = (image * 255.0).clip(0, 255).astype(np.uint8)
    else:
        display_img = image.copy()

    # Determine if input was RGB
    # We will output standard BGR for OpenCV saving
    h, w = display_img.shape[:2]

    # Overlay banner height proportional to image
    banner_height = max(60, int(h * 0.22))
    overlay = display_img.copy()

    # Draw semi-transparent dark banner at bottom
    cv2.rectangle(
        overlay,
        (0, h - banner_height),
        (w, h),
        config.CV_BANNER_COLOR,
        -1,
    )
    # Blend banner with 75% opacity
    cv2.addWeighted(overlay, 0.78, display_img, 0.22, 0, display_img)

    # Accent decorative line
    cv2.line(
        display_img,
        (0, h - banner_height),
        (w, h - banner_height),
        config.CV_ACCENT_COLOR,
        2,
    )

    # Header badge: Model Tag + Confidence
    badge_text = f"[{model_tag}]"
    if confidence is not None:
        badge_text += f" Score: {confidence:.2f}"

    cv2.putText(
        display_img,
        badge_text,
        (15, h - banner_height + 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        config.CV_ACCENT_COLOR,
        1,
        cv2.LINE_AA,
    )

    # Word-wrap caption text to fit width
    words = caption.split()
    lines: List[str] = []
    current_line = []
    max_text_width = w - 30

    font_scale = 0.58 if w < 400 else 0.68
    thickness = 1 if w < 400 else 2

    for word in words:
        test_line = " ".join(current_line + [word])
        (text_w, _), _ = cv2.getTextSize(
            test_line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        if text_w <= max_text_width:
            current_line.append(word)
        else:
            if current_line:
                lines.append(" ".join(current_line))
            current_line = [word]
    if current_line:
        lines.append(" ".join(current_line))

    # Render wrapped lines
    y_start = h - banner_height + 48
    line_spacing = 24
    for i, line in enumerate(lines[:3]):  # Render up to 3 lines
        y_pos = y_start + (i * line_spacing)
        if y_pos > h - 8:
            break
        # Text shadow for readability
        cv2.putText(
            display_img,
            line,
            (16, y_pos + 1),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            thickness + 1,
            cv2.LINE_AA,
        )
        cv2.putText(
            display_img,
            line,
            (15, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            config.CV_TEXT_COLOR,
            thickness,
            cv2.LINE_AA,
        )

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), display_img)

    return display_img


def visualize_attention_cv(
    image_bgr: np.ndarray,
    words: List[str],
    attention_weights: np.ndarray,
    output_path: Optional[Union[str, Path]] = None,
) -> np.ndarray:
    """Generate side-by-side attention heatmaps overlaid on the image for each word.

    Args:
        image_bgr: Original image in BGR format.
        words: List of predicted words (length T).
        attention_weights: Array of shape (T, 49) or (T, 7, 7) attention probabilities.
        output_path: Optional path to save the combined image grid.

    Returns:
        np.ndarray: Combined image montage displaying attention for each token.
    """
    t_steps = len(words)
    if t_steps == 0:
        return image_bgr

    h, w = image_bgr.shape[:2]
    tiles = []

    for t in range(t_steps):
        word = words[t]
        attn = attention_weights[t]

        # Reshape to 7x7 spatial grid if flat
        if attn.ndim == 1:
            side = int(np.sqrt(attn.shape[0]))
            attn_grid = attn.reshape(side, side)
        else:
            attn_grid = attn

        # Normalize attention map to [0, 255]
        attn_norm = (attn_grid - attn_grid.min()) / (attn_grid.max() - attn_grid.min() + 1e-8)
        attn_uint8 = (attn_norm * 255).astype(np.uint8)

        # Upscale attention map to image resolution using cubic interpolation
        heatmap = cv2.resize(attn_uint8, (w, h), interpolation=cv2.INTER_CUBIC)
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        # Blend original image and attention heatmap
        blended = cv2.addWeighted(image_bgr, 0.65, heatmap_color, 0.35, 0)

        # Draw token label header banner
        cv2.rectangle(blended, (0, 0), (w, 32), (20, 20, 20), -1)
        cv2.putText(
            blended,
            f"Step {t+1}: {word}",
            (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        tiles.append(blended)

    # Assemble into a clean montage grid
    cols = min(4, len(tiles))
    rows = (len(tiles) + cols - 1) // cols

    tile_h, tile_w = tiles[0].shape[:2]
    montage = np.zeros((rows * tile_h, cols * tile_w, 3), dtype=np.uint8)

    for idx, tile in enumerate(tiles):
        r = idx // cols
        c = idx % cols
        montage[r * tile_h : (r + 1) * tile_h, c * tile_w : (c + 1) * tile_w] = tile

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), montage)

    return montage


def create_synthetic_dataset(
    output_dir: Union[str, Path] = config.SAMPLES_DIR,
) -> List[dict]:
    """Generate synthetic realistic test images using OpenCV shapes and patterns.

    Enables testing the entire image captioning pipeline instantly without requiring
    external downloads.

    Returns:
        List of dicts with 'image_path' and 'captions'.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = [
        {
            "filename": "dog_park.jpg",
            "captions": [
                "a brown dog playing on the green grass",
                "a playful puppy running across the lawn",
                "a cute dog enjoying the sunny park outdoors",
            ],
            "draw": "dog",
        },
        {
            "filename": "red_car.jpg",
            "captions": [
                "a red sports car driving on the asphalt road",
                "a bright red automobile on the highway",
                "a modern red car parked along the street",
            ],
            "draw": "car",
        },
        {
            "filename": "cat_window.jpg",
            "captions": [
                "a ginger cat resting peacefully near the window",
                "a cute kitten sitting on a cozy rug",
                "a sleepy cat looking out at the outdoors",
            ],
            "draw": "cat",
        },
        {
            "filename": "sunset_ocean.jpg",
            "captions": [
                "a golden sunset over the calm ocean waves",
                "beautiful colorful evening sky above the sea",
                "a peaceful sunset landscape with orange sky and water",
            ],
            "draw": "sunset",
        },
        {
            "filename": "bicycle_tree.jpg",
            "captions": [
                "a vintage bicycle parked next to a green tree",
                "a city bike standing by the park sidewalk",
                "a commuter bicycle resting under a large tree",
            ],
            "draw": "bicycle",
        },
    ]

    records = []

    for item in samples:
        img_path = out_dir / item["filename"]
        canvas = np.zeros((300, 400, 3), dtype=np.uint8)

        theme = item["draw"]
        if theme == "dog":
            # Sky & Grass
            canvas[:150] = (235, 206, 135)   # Light blue sky
            canvas[150:] = (45, 139, 34)     # Forest green grass
            # Sun
            cv2.circle(canvas, (350, 50), 30, (0, 215, 255), -1)
            # Dog body (brown oval)
            cv2.ellipse(canvas, (200, 210), (55, 32), 0, 0, 360, (28, 70, 139), -1)
            # Dog head
            cv2.circle(canvas, (250, 185), 24, (28, 70, 139), -1)
            # Dog ears
            cv2.ellipse(canvas, (262, 175), (8, 16), 25, 0, 360, (19, 45, 92), -1)
            # Eye & nose
            cv2.circle(canvas, (258, 182), 3, (0, 0, 0), -1)
            cv2.circle(canvas, (270, 188), 4, (0, 0, 0), -1)
            # Legs
            cv2.line(canvas, (165, 225), (165, 260), (28, 70, 139), 8)
            cv2.line(canvas, (185, 225), (185, 260), (28, 70, 139), 8)
            cv2.line(canvas, (225, 225), (225, 260), (28, 70, 139), 8)
            cv2.line(canvas, (240, 225), (240, 260), (28, 70, 139), 8)

        elif theme == "car":
            # Sky & Road
            canvas[:180] = (220, 220, 220)   # Light gray sky/background
            canvas[180:] = (60, 60, 60)      # Dark asphalt road
            # Road dashed stripe
            for rx in range(0, 400, 50):
                cv2.line(canvas, (rx, 250), (rx + 30, 250), (255, 255, 255), 4)
            # Car body (Red)
            cv2.rectangle(canvas, (110, 170), (290, 215), (30, 30, 220), -1)
            # Car cabin
            pts = np.array([[145, 170], [175, 135], [235, 135], [260, 170]], np.int32)
            cv2.fillPoly(canvas, [pts], (30, 30, 220))
            # Windows
            win_pts = np.array([[155, 168], [178, 140], [230, 140], [250, 168]], np.int32)
            cv2.fillPoly(canvas, [win_pts], (240, 240, 240))
            # Wheels
            cv2.circle(canvas, (155, 218), 22, (20, 20, 20), -1)
            cv2.circle(canvas, (155, 218), 10, (180, 180, 180), -1)
            cv2.circle(canvas, (250, 218), 22, (20, 20, 20), -1)
            cv2.circle(canvas, (250, 218), 10, (180, 180, 180), -1)

        elif theme == "cat":
            # Indoor wall and floor
            canvas[:170] = (210, 225, 235)   # Cozy wall
            canvas[170:] = (90, 120, 160)    # Carpet floor
            # Window frame
            cv2.rectangle(canvas, (40, 25), (150, 150), (255, 255, 255), -1)
            cv2.rectangle(canvas, (40, 25), (150, 150), (100, 100, 100), 3)
            cv2.line(canvas, (95, 25), (95, 150), (100, 100, 100), 2)
            cv2.line(canvas, (40, 87), (150, 87), (100, 100, 100), 2)
            # Cat body (Ginger/Orange)
            cv2.ellipse(canvas, (250, 220), (45, 30), -15, 0, 360, (40, 120, 240), -1)
            # Cat head
            cv2.circle(canvas, (215, 190), 22, (40, 120, 240), -1)
            # Ears
            ear_l = np.array([[200, 180], [205, 160], [215, 175]], np.int32)
            ear_r = np.array([[220, 175], [230, 160], [235, 180]], np.int32)
            cv2.fillPoly(canvas, [ear_l], (30, 100, 220))
            cv2.fillPoly(canvas, [ear_r], (30, 100, 220))
            # Tail
            cv2.ellipse(canvas, (300, 220), (25, 10), 35, 0, 360, (40, 120, 240), 6)

        elif theme == "sunset":
            # Gradient sky (Orange to Magenta to Blue)
            for y in range(160):
                ratio = y / 160.0
                b = int(30 + ratio * 15)
                g = int(80 + (1 - ratio) * 120)
                r = int(240 - ratio * 40)
                canvas[y, :] = (b, g, r)
            # Sea (Navy with reflections)
            for y in range(160, 300):
                canvas[y, :] = (120, 60, 20)
            # Sun sinking
            cv2.circle(canvas, (200, 160), 35, (50, 210, 255), -1)
            # Water reflections
            for ry in range(170, 280, 15):
                rw = int(60 + (ry - 170) * 1.2)
                cv2.line(canvas, (200 - rw // 2, ry), (200 + rw // 2, ry), (80, 180, 255), 2)

        elif theme == "bicycle":
            # Park scene
            canvas[:170] = (240, 240, 220)
            canvas[170:] = (80, 160, 80)
            # Tree trunk & leaves
            cv2.rectangle(canvas, (50, 80), (80, 260), (30, 50, 90), -1)
            cv2.circle(canvas, (65, 80), 55, (40, 120, 40), -1)
            # Bicycle wheels
            cv2.circle(canvas, (190, 225), 32, (30, 30, 30), 4)
            cv2.circle(canvas, (285, 225), 32, (30, 30, 30), 4)
            # Bike frame
            cv2.line(canvas, (190, 225), (230, 225), (200, 50, 50), 4)
            cv2.line(canvas, (230, 225), (275, 185), (200, 50, 50), 4)
            cv2.line(canvas, (190, 225), (225, 185), (200, 50, 50), 4)
            cv2.line(canvas, (225, 185), (275, 185), (200, 50, 50), 4)
            cv2.line(canvas, (230, 225), (225, 175), (200, 50, 50), 4)
            cv2.line(canvas, (285, 225), (275, 175), (200, 50, 50), 4)
            # Handlebars and seat
            cv2.line(canvas, (265, 170), (285, 170), (20, 20, 20), 4)
            cv2.line(canvas, (215, 175), (235, 175), (20, 20, 20), 5)

        cv2.imwrite(str(img_path), canvas)
        records.append({
            "image_path": str(img_path),
            "captions": item["captions"],
        })

    return records
