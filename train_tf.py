"""Training Script for TensorFlow / Keras Image Captioning with Additive Attention.

Uses tf.data with OpenCV decoding, custom training step with Teacher Forcing,
and saves model weights.
"""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("KERAS_HOME", os.path.expanduser("/tmp/.keras"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf
from tqdm import tqdm

import config
from src.dataset import create_tf_dataset, load_captions_file
from src.models.tf_captioner import TFCaptioner
from src.opencv_utils import create_synthetic_dataset
from src.vocab import Vocabulary


def parse_args():
    parser = argparse.ArgumentParser(description="Train TensorFlow Image Captioner")
    parser.add_argument(
        "--data_file",
        type=str,
        default=str(config.ANNOTATIONS_FILE),
        help="Path to captions JSON or text file",
    )
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE, help="Learning rate")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=str(config.TF_CHECKPOINT_DIR),
        help="Directory to save TensorFlow model weights",
    )
    parser.add_argument(
        "--vocab_path",
        type=str,
        default=str(config.VOCAB_FILE),
        help="Path to save/load vocabulary JSON",
    )
    return parser.parse_args()


def train():
    args = parse_args()
    print("[TensorFlow] Initializing training...")

    # 1. Dataset check
    data_path = Path(args.data_file)
    if not data_path.exists():
        print(f"Data file '{data_path}' not found. Generating OpenCV synthetic dataset...")
        records = create_synthetic_dataset(config.SAMPLES_DIR)
        with open(data_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)

    pairs = load_captions_file(data_path)
    print(f"Loaded {len(pairs)} image-caption pairs.")

    # 2. Vocabulary
    vocab_path = Path(args.vocab_path)
    if vocab_path.exists():
        print(f"Loading vocabulary from {vocab_path}")
        vocab = Vocabulary.load(vocab_path)
    else:
        print("Building vocabulary from captions...")
        vocab = Vocabulary(freq_threshold=config.FREQ_THRESHOLD)
        vocab.build_vocabulary([p[1] for p in pairs])
        vocab.save(vocab_path)
        print(f"Saved vocabulary ({len(vocab)} words) to {vocab_path}")

    # 3. Create tf.data.Dataset
    train_dataset = create_tf_dataset(
        pairs,
        vocab,
        batch_size=args.batch_size,
        is_train=True,
        max_len=config.MAX_CAPTION_LEN,
    )

    # 4. Model & Optimizer
    model = TFCaptioner(
        vocab_size=len(vocab),
        embedding_dim=config.EMBED_DIM,
        units=config.DECODER_DIM,
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)

    # 5. Training Loop
    print("\nStarting TensorFlow Training Loop...")
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        batch_count = 0

        for img_tensor, target in tqdm(train_dataset, desc=f"Epoch {epoch}/{args.epochs}"):
            loss = model.train_step(img_tensor, target, optimizer)
            total_loss += float(loss.numpy())
            batch_count += 1

        avg_loss = total_loss / max(1, batch_count)
        print(f"Epoch [{epoch}/{args.epochs}] - Average Loss: {avg_loss:.4f}")

    # Save model weights
    save_weights_path = str(ckpt_dir / "tf_captioner_weights.weights.h5")
    try:
        model.save_weights(save_weights_path)
        print(f"--> Saved TensorFlow weights to {save_weights_path}")
    except Exception as e:
        # Fallback format if needed
        save_weights_path = str(ckpt_dir / "tf_captioner_weights")
        model.save_weights(save_weights_path)
        print(f"--> Saved TensorFlow weights to {save_weights_path}")

    print("\nTensorFlow training completed successfully!")


if __name__ == "__main__":
    train()
