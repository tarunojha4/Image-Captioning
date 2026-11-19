"""Flask Web Application for Interactive Deep Learning Image Captioning Studio.

Provides an interactive web dashboard and REST API for:
- Uploading images or selecting sample images
- Selecting between PyTorch, TensorFlow, or Cross-Framework comparison
- Viewing OpenCV styled HUD captioned images and Attention Heatmap grids
"""

import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("KERAS_HOME", os.path.expanduser("/tmp/.keras"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/.matplotlib")

import cv2
from flask import Flask, jsonify, render_template, request, send_from_directory
import numpy as np
import torch

import config
from predict import load_pytorch_model, load_tf_model
from src.bridge import compare_frameworks
from src.opencv_utils import (
    load_image_cv,
    normalize_image,
    overlay_caption_on_image,
    visualize_attention_cv,
)
from src.vocab import Vocabulary

app = Flask(__name__, template_folder="templates")

# Global model cache for fast web inference
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
vocab = None
pt_model = None
tf_model = None


def get_models():
    global vocab, pt_model, tf_model
    if vocab is None:
        if not config.VOCAB_FILE.exists():
            from src.opencv_utils import create_synthetic_dataset
            records = create_synthetic_dataset(config.SAMPLES_DIR)
            import json
            with open(config.ANNOTATIONS_FILE, "w", encoding="utf-8") as f:
                json.dump(records, f, indent=2)
            v = Vocabulary()
            v.build_vocabulary([cap for r in records for cap in r["captions"]])
            v.save(config.VOCAB_FILE)

        vocab = Vocabulary.load(config.VOCAB_FILE)

    if pt_model is None:
        print("[Web App] Loading PyTorch model into memory...")
        pt_model = load_pytorch_model(len(vocab), str(config.PYTORCH_CHECKPOINT), device)

    if tf_model is None:
        print("[Web App] Loading TensorFlow model into memory...")
        import tensorflow as tf
        tf_model = load_tf_model(len(vocab), config.TF_CHECKPOINT_DIR)

    return vocab, pt_model, tf_model


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/data/samples/<path:filename>")
def serve_sample(filename):
    return send_from_directory(config.SAMPLES_DIR, filename)


@app.route("/outputs/<path:filename>")
def serve_output(filename):
    return send_from_directory(config.OUTPUT_DIR, filename)


@app.route("/caption", methods=["POST"])
def generate_caption_endpoint():
    v, pm, tm = get_models()

    # Determine input image
    temp_dir = config.OUTPUT_DIR / "uploads"
    temp_dir.mkdir(parents=True, exist_ok=True)

    if "file" in request.files and request.files["file"].filename != "":
        file = request.files["file"]
        image_path = temp_dir / file.filename
        file.save(str(image_path))
    elif "sample" in request.form and request.form["sample"] != "":
        sample_name = request.form["sample"]
        image_path = config.SAMPLES_DIR / sample_name
    else:
        # Default fallback
        image_path = config.SAMPLES_DIR / "dog_park.jpg"

    framework = request.form.get("framework", "pytorch")
    method = request.form.get("method", "beam")

    t_stamp = int(time.time() * 1000)
    out_img_name = f"web_{t_stamp}_caption.jpg"
    out_img_path = config.OUTPUT_DIR / out_img_name

    heatmap_name = f"web_{t_stamp}_heatmap.jpg"
    heatmap_path = config.OUTPUT_DIR / heatmap_name

    if framework == "both":
        results = compare_frameworks(
            image_path=image_path,
            pytorch_model=pm,
            tf_model=tm,
            vocab=v,
            device=device,
            output_path=out_img_path,
        )
        return jsonify({
            "caption": f"PyTorch: \"{results['pytorch_caption']}\" | TensorFlow: \"{results['tensorflow_caption']}\"",
            "framework": "Dual (PyTorch & TensorFlow)",
            "latency_ms": f"{results['pytorch_latency_ms']} ms (PT) / {results['tensorflow_latency_ms']} ms (TF)",
            "image_url": f"/outputs/{out_img_name}",
            "heatmap_url": None,
        })

    # Prepare image
    img_rgb = load_image_cv(image_path, target_size=config.IMAGE_SIZE, to_rgb=True)
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    img_norm = normalize_image(img_rgb)
    t0 = time.perf_counter()

    if framework == "pytorch":
        img_tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).float().to(device)
        caption, alphas, score = pm.generate(img_tensor, v, method=method, beam_width=3)
        latency = (time.perf_counter() - t0) * 1000.0

        overlay_caption_on_image(
            image=img_bgr,
            caption=caption,
            model_tag=f"PyTorch ({method.upper()})",
            output_path=out_img_path,
        )

        words = caption.split()
        alphas_np = alphas.cpu().numpy()
        heatmap_url = None
        if len(alphas_np) > 0 and len(words) > 0:
            visualize_attention_cv(
                image_bgr=img_bgr,
                words=words[: len(alphas_np)],
                attention_weights=alphas_np,
                output_path=heatmap_path,
            )
            heatmap_url = f"/outputs/{heatmap_name}"

    else:  # tensorflow
        caption, alphas = tm.predict_caption(img_norm, v)
        latency = (time.perf_counter() - t0) * 1000.0

        overlay_caption_on_image(
            image=img_bgr,
            caption=caption,
            model_tag="TensorFlow MobileNetV2",
            output_path=out_img_path,
        )

        words = caption.split()
        heatmap_url = None
        if len(alphas) > 0 and len(words) > 0:
            visualize_attention_cv(
                image_bgr=img_bgr,
                words=words[: len(alphas)],
                attention_weights=alphas,
                output_path=heatmap_path,
            )
            heatmap_url = f"/outputs/{heatmap_name}"

    return jsonify({
        "caption": caption,
        "framework": framework,
        "latency_ms": round(latency, 2),
        "image_url": f"/outputs/{out_img_name}",
        "heatmap_url": heatmap_url,
    })


def main():
    parser = argparse.ArgumentParser(description="Start Image Captioning Web Studio")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host interface")
    parser.add_argument("--port", type=int, default=5000, help="Port number")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()

    # Preload models
    get_models()

    print(f"\n===========================================================")
    print(f"  AI Image Captioning Studio Web Application is Ready!")
    print(f"  Open your browser at: http://{args.host}:{args.port}")
    print(f"===========================================================\n")

    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
