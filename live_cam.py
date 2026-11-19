"""Real-time OpenCV Webcam / Video Stream Captioning HUD.

Captures frames from a camera or video file, runs caption inference periodically
to ensure smooth 30 FPS display, and renders a live styled HUD banner.

Controls:
    'q' : Quit
    's' : Save current snapshot with caption
    'm' : Toggle model (PyTorch <-> TensorFlow)
"""

import argparse
import os
import time
from pathlib import Path
import cv2
import numpy as np
import torch

os.environ.setdefault("KERAS_HOME", os.path.expanduser("/tmp/.keras"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import config
from predict import load_pytorch_model
from src.models.pytorch_captioner import PyTorchCaptioner
from src.models.tf_captioner import TFCaptioner
from src.opencv_utils import normalize_image, overlay_caption_on_image
from src.vocab import Vocabulary


def parse_args():
    parser = argparse.ArgumentParser(description="Live OpenCV Camera Captioning HUD")
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Camera device index (0) or path to video file",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=30,
        help="Run inference every N frames to maintain high video FPS",
    )
    parser.add_argument(
        "--framework",
        type=str,
        choices=["pytorch", "tensorflow"],
        default="pytorch",
        help="Initial framework",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    source = int(args.source) if args.source.isdigit() else args.source

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Unable to open video source '{source}'.")
        print("Note: If running on a headless machine or without a physical webcam, pass a video file: --source video.mp4")
        return

    # Load Vocabulary
    vocab_path = config.VOCAB_FILE
    if not vocab_path.exists():
        print("Error: Vocabulary not found. Run demo_quickstart.py or train_pytorch.py first.")
        return
    vocab = Vocabulary.load(vocab_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    # Load PyTorch Model
    print("Loading PyTorch model...")
    pt_model = load_pytorch_model(len(vocab), str(config.PYTORCH_CHECKPOINT), device)

    # Load TensorFlow Model
    print("Loading TensorFlow model...")
    import tensorflow as tf
    tf_model = TFCaptioner(vocab_size=len(vocab))
    dummy_img = tf.zeros((1, 224, 224, 3))
    dummy_feat = tf_model.encoder(dummy_img)
    dummy_hidden = tf_model.decoder.reset_state(1)
    dummy_in = tf.zeros((1, 1), dtype=tf.int32)
    tf_model.decoder(dummy_in, dummy_feat, dummy_hidden)

    active_framework = args.framework
    current_caption = "Initializing deep learning captioning..."
    inference_ms = 0.0

    frame_idx = 0
    t_start = time.time()
    fps = 0.0

    print("\nLive Stream Captioning started.")
    print("Controls: [q] Quit  |  [s] Save Frame  |  [m] Toggle PyTorch/TensorFlow\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1

            # Run inference periodically
            if frame_idx % args.interval == 1 or frame_idx == 1:
                # Prepare image with OpenCV
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                resized_rgb = cv2.resize(frame_rgb, config.IMAGE_SIZE)
                img_norm = normalize_image(resized_rgb)

                t0 = time.perf_counter()
                if active_framework == "pytorch":
                    img_tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).float().to(device)
                    caption, _, _ = pt_model.generate(img_tensor, vocab, method="greedy", max_len=18)
                else:
                    caption, _ = tf_model.predict_caption(img_norm, vocab, max_len=18)

                inference_ms = (time.perf_counter() - t0) * 1000.0
                current_caption = caption

            # Calculate FPS
            if frame_idx % 15 == 0:
                fps = 15.0 / (time.time() - t_start + 1e-6)
                t_start = time.time()

            # Render OpenCV HUD overlay
            hud_tag = f"{active_framework.upper()} ({fps:.1f} FPS | {inference_ms:.0f} ms)"
            display_frame = overlay_caption_on_image(
                image=frame,
                caption=current_caption,
                model_tag=hud_tag,
            )

            cv2.imshow("Real-Time OpenCV Image Captioning", display_frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key == ord("s"):
                snap_path = config.OUTPUT_DIR / f"snapshot_{int(time.time())}.jpg"
                cv2.imwrite(str(snap_path), display_frame)
                print(f"Snapshot saved to: {snap_path}")
            elif key == ord("m"):
                active_framework = "tensorflow" if active_framework == "pytorch" else "pytorch"
                print(f"Switched framework to: {active_framework}")

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
