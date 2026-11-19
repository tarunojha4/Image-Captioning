"""Training Script for PyTorch Image Captioning with Bahdanau Attention.

Supports custom datasets (Flickr8k/Flickr30k/JSON), gradient clipping,
checkpoint saving, and validation BLEU scoring.
"""

import argparse
import os
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import config
from src.dataset import CaptionCollate, PyTorchCaptionDataset, load_captions_file
from src.metrics import compute_bleu_scores
from src.models.pytorch_captioner import PyTorchCaptioner
from src.opencv_utils import create_synthetic_dataset
from src.vocab import Vocabulary


def parse_args():
    parser = argparse.ArgumentParser(description="Train PyTorch Image Captioner")
    parser.add_argument(
        "--data_file",
        type=str,
        default=str(config.ANNOTATIONS_FILE),
        help="Path to captions JSON or text file",
    )
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE, help="Learning rate")
    parser.add_argument(
        "--save_path",
        type=str,
        default=str(config.PYTORCH_CHECKPOINT),
        help="Path to save model checkpoint",
    )
    parser.add_argument(
        "--vocab_path",
        type=str,
        default=str(config.VOCAB_FILE),
        help="Path to save/load vocabulary JSON",
    )
    parser.add_argument(
        "--use_synthetic_data",
        action="store_true",
        help="Generate synthetic OpenCV dataset if data file is missing",
    )
    return parser.parse_args()


def train():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[PyTorch] Using device: {device}")

    # 1. Check or generate dataset
    data_path = Path(args.data_file)
    if not data_path.exists():
        print(f"Data file '{data_path}' not found. Generating OpenCV synthetic dataset...")
        records = create_synthetic_dataset(config.SAMPLES_DIR)
        import json
        with open(data_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)
        print(f"Saved synthetic dataset to {data_path}")

    # Load pairs
    pairs = load_captions_file(data_path)
    print(f"Loaded {len(pairs)} image-caption pairs.")

    # 2. Build or load vocabulary
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

    # 3. Create Dataset and DataLoader
    dataset = PyTorchCaptionDataset(pairs, vocab, is_train=True, augment=True)
    val_size = max(1, int(0.15 * len(dataset)))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    collate_fn = CaptionCollate(pad_idx=config.PAD_IDX)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=False,
    )

    # 4. Initialize Model, Criterion, Optimizer
    model = PyTorchCaptioner(
        vocab_size=len(vocab),
        embed_dim=config.EMBED_DIM,
        decoder_dim=config.DECODER_DIM,
        encoder_dim=config.ENCODER_DIM,
        attention_dim=config.ATTENTION_DIM,
        fine_tune_encoder=config.FINE_TUNE_ENCODER,
    ).to(device)

    # Ignore loss on PAD tokens
    criterion = nn.CrossEntropyLoss(ignore_index=config.PAD_IDX)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
    )

    # 5. Training Loop
    print("\nStarting PyTorch Model Training...")
    best_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0

        for images, captions, lengths in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            images = images.to(device)
            captions = captions.to(device)
            lengths = lengths.to(device)

            optimizer.zero_grad()
            # Predictions: (B, max_len - 1, vocab_size)
            predictions, alphas = model(images, captions, lengths)

            # Targets are captions shifted by 1 (predicting w_{t+1})
            targets = captions[:, 1:]

            # Flatten for cross-entropy: (B * T, vocab_size) vs (B * T)
            loss = criterion(
                predictions.reshape(-1, predictions.size(-1)),
                targets.reshape(-1),
            )

            # Doubly stochastic attention regularization (encourages model to attend to every pixel)
            # sum_t (1 - sum_pixel alpha_{t, p})^2
            att_reg = ((1.0 - alphas.sum(dim=1)) ** 2).mean()
            loss = loss + 1.0 * att_reg

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.CLIP_GRAD_NORM)
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / max(1, len(train_loader))
        print(f"Epoch [{epoch}/{args.epochs}] - Training Loss: {epoch_loss:.4f}")

        # Quick validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, captions, lengths in val_loader:
                images, captions = images.to(device), captions.to(device)
                predictions, alphas = model(images, captions, lengths)
                targets = captions[:, 1:]
                v_loss = criterion(
                    predictions.reshape(-1, predictions.size(-1)),
                    targets.reshape(-1),
                )
                val_loss += v_loss.item()
        val_loss /= max(1, len(val_loader))
        print(f"Validation Loss: {val_loss:.4f}")

        # Save checkpoint
        if val_loss < best_loss or epoch == args.epochs:
            best_loss = val_loss
            Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "vocab_size": len(vocab),
                    "embed_dim": config.EMBED_DIM,
                    "decoder_dim": config.DECODER_DIM,
                    "encoder_dim": config.ENCODER_DIM,
                    "attention_dim": config.ATTENTION_DIM,
                    "loss": val_loss,
                },
                args.save_path,
            )
            print(f"--> Saved checkpoint to {args.save_path}")

    print("\nTraining completed successfully!")


if __name__ == "__main__":
    train()
