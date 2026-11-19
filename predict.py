"""Inference CLI for Image Captioning with OpenCV Visual Overlays and Attention Heatmaps.

Usage:
    python predict.py --image data/samples/dog_park.jpg --framework pytorch --attention
    python predict.py --image data/samples/red_car.jpg --framework both
"""

import argparse
import os
from pathlib import Path
import cv2
import numpy as np
import torch

os.environ.setdefault("KERAS_HOME", os.path.expanduser("/tmp/.keras"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import config
from src.bridge import compare_frameworks
from src.models.pytorch_captioner import PyTorchCaptioner
from src.models.tf_captioner import TFCaptioner
from src.opencv_utils import (
    load_image_cv,
    normalize_image,
    overlay_caption_on_image,
    visualize_attention_cv,
)
from src.vocab import Vocabulary


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Caption for an Image")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument(
        "--framework",
        type=str,
        choices=["pytorch", "tensorflow", "both"],
        default="pytorch",
        help="Deep learning framework to use",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["greedy", "beam"],
        default="beam",
        help="PyTorch decoding method",
    )
    parser.add_argument("--beam_width", type=int, default=3, help="Beam width for beam search")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save overlaid visualization image",
    )
    parser.add_argument(
        "--attention",
        action="store_true",
        help="Generate and save attention heatmap visualization",
    )
    parser.add_argument(
        "--vocab_path",
        type=str,
        default=str(config.VOCAB_FILE),
        help="Path to vocabulary JSON",
    )
    parser.add_argument(
        "--pt_checkpoint",
        type=str,
        default=str(config.PYTORCH_CHECKPOINT),
        help="Path to PyTorch checkpoint",
    )
    return parser.parse_args()


def load_pytorch_model(vocab_size: int, checkpoint_path: str, device: torch.device):
    ckpt = Path(checkpoint_path)
    if ckpt.exists():
        print(f"Loading PyTorch checkpoint from {ckpt}...")
        checkpoint = torch.load(ckpt, map_location=device)
        embed_dim = checkpoint.get("embed_dim", config.EMBED_DIM)
        decoder_dim = checkpoint.get("decoder_dim", config.DECODER_DIM)
        encoder_dim = checkpoint.get("encoder_dim", config.ENCODER_DIM)
        attention_dim = checkpoint.get("attention_dim", config.ATTENTION_DIM)

        model = PyTorchCaptioner(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            decoder_dim=decoder_dim,
            encoder_dim=encoder_dim,
            attention_dim=attention_dim,
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        print(f"Checkpoint '{ckpt}' not found. Initialized with fresh weights.")
        model = PyTorchCaptioner(vocab_size=vocab_size).to(device)
    model.eval()
    return model


def load_tf_model(vocab_size: int, ckpt_dir: Path):
    import tensorflow as tf
    model = TFCaptioner(vocab_size=vocab_size)
    # Build model by calling on dummy input
    dummy_img = tf.zeros((1, 224, 224, 3))
    dummy_feat = model.encoder(dummy_img)
    dummy_hidden = model.decoder.reset_state(1)
    dummy_in = tf.zeros((1, 1), dtype=tf.int32)
    model.decoder(dummy_in, dummy_feat, dummy_hidden)

    weights_h5 = ckpt_dir / "tf_captioner_weights.weights.h5"
    if weights_h5.exists():
        print(f"Loading TensorFlow weights from {weights_h5}...")
        model.load_weights(str(weights_h5))
    return model


def main():
    args = parse_args()
    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found at: {image_path}")

    # Load Vocabulary
    vocab_path = Path(args.vocab_path)
    if not vocab_path.exists():
        raise FileNotFoundError(
            f"Vocabulary file not found at {vocab_path}. Run train_pytorch.py or demo_quickstart.py first."
        )
    vocab = Vocabulary.load(vocab_path)
    print(f"Vocabulary loaded ({len(vocab)} words).")

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    # Output paths
    stem = image_path.stem
    default_out = config.OUTPUT_DIR / f"{stem}_caption.jpg"
    out_path = Path(args.output) if args.output else default_out

    if args.framework == "both":
        print(f"\n--- Running Cross-Framework Comparison (PyTorch & TensorFlow) ---")
        pt_model = load_pytorch_model(len(vocab), args.pt_checkpoint, device)
        tf_model = load_tf_model(len(vocab), config.TF_CHECKPOINT_DIR)

        results = compare_frameworks(
            image_path=image_path,
            pytorch_model=pt_model,
            tf_model=tf_model,
            vocab=vocab,
            device=device,
            output_path=out_path,
        )
        print(f"[PyTorch Result]    : {results['pytorch_caption']} ({results['pytorch_latency_ms']} ms)")
        print(f"[TensorFlow Result] : {results['tensorflow_caption']} ({results['tensorflow_latency_ms']} ms)")
        print(f"[OpenCV Output]     : Saved comparison HUD visualization to {out_path}")
        return

    # Load image for processing
    img_rgb = load_image_cv(image_path, target_size=config.IMAGE_SIZE, to_rgb=True)
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    img_norm = normalize_image(img_rgb)

    if args.framework == "pytorch":
        print(f"\n--- PyTorch Caption Generation ({args.method.upper()}) ---")
        pt_model = load_pytorch_model(len(vocab), args.pt_checkpoint, device)
        img_tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).float().to(device)

        caption, alphas, score = pt_model.generate(
            img_tensor,
            vocab,
            method=args.method,
            beam_width=args.beam_width,
            max_len=config.MAX_CAPTION_LEN,
        )
        model_tag = f"PyTorch ResNet50 ({args.method})"
        conf = float(np.exp(score)) if score is not None and score < 0 else None

    else:  # tensorflow
        print(f"\n--- TensorFlow / Keras Caption Generation ---")
        tf_model = load_tf_model(len(vocab), config.TF_CHECKPOINT_DIR)
        caption, alphas = tf_model.predict_caption(img_norm, vocab)
        model_tag = "TensorFlow MobileNetV2"
        conf = None

    print(f"Generated Caption: \"{caption}\"")

    # Render OpenCV Caption Banner HUD
    overlay_img = overlay_caption_on_image(
        image=img_bgr,
        caption=caption,
        model_tag=model_tag,
        confidence=conf,
        output_path=out_path,
    )
    print(f"Saved captioned image to: {out_path}")

    # Generate Attention Heatmap Grid if requested
    if args.attention:
        words = caption.split()
        attn_np = alphas.cpu().numpy() if isinstance(alphas, torch.Tensor) else alphas
        if len(attn_np) > 0 and len(words) > 0:
            attn_out = config.OUTPUT_DIR / f"{stem}_attention_heatmap.jpg"
            visualize_attention_cv(
                image_bgr=img_bgr,
                words=words[: len(attn_np)],
                attention_weights=attn_np,
                output_path=attn_out,
            )
            print(f"Saved attention heatmap grid to: {attn_out}")


if __name__ == "__main__":
    import tensorflow as tf
    main()
