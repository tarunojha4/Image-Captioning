"""Central Configuration for Image Captioning Project.

Handles paths, hyper-parameters, image dimensions, and environment settings
across both PyTorch and TensorFlow pipelines.
"""

import os
from pathlib import Path

# Fix Keras & TensorFlow & Matplotlib directory permissions in sandbox / restricted environments
os.environ.setdefault("KERAS_HOME", os.path.expanduser("/tmp/.keras"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/.matplotlib")

# Base directory
BASE_DIR = Path(__file__).resolve().parent

# Data directories
DATA_DIR = BASE_DIR / "data"
SAMPLES_DIR = DATA_DIR / "samples"
ANNOTATIONS_FILE = DATA_DIR / "captions.json"
VOCAB_FILE = DATA_DIR / "vocab.json"
OUTPUT_DIR = BASE_DIR / "outputs"

# Model Checkpoints
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
PYTORCH_CHECKPOINT = CHECKPOINT_DIR / "best_model_pytorch.pth"
TF_CHECKPOINT_DIR = CHECKPOINT_DIR / "tf_model"

# Create directories if they do not exist
for directory in [DATA_DIR, SAMPLES_DIR, OUTPUT_DIR, CHECKPOINT_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Image Preprocessing (OpenCV / CNN Input)
IMAGE_SIZE = (224, 224)          # (Height, Width) for CNN models
IMAGE_CHANNELS = 3
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Vocabulary & Tokenizer
SPECIAL_TOKENS = {
    "pad_token": "<pad>",
    "start_token": "<start>",
    "end_token": "<end>",
    "unk_token": "<unk>",
}
PAD_IDX = 0
START_IDX = 1
END_IDX = 2
UNK_IDX = 3
FREQ_THRESHOLD = 1               # Minimum word occurrence frequency
MAX_CAPTION_LEN = 25             # Maximum token sequence length

# Model Hyperparameters
ENCODER_DIM = 2048               # ResNet-50 feature dimension (or 1280 for MobileNetV2)
ATTENTION_DIM = 256              # Bahdanau attention intermediate dimension
EMBED_DIM = 256                  # Word embedding dimension
DECODER_DIM = 512                # LSTM/GRU hidden dimension
DROPOUT = 0.3
FINE_TUNE_ENCODER = False

# Training Defaults
BATCH_SIZE = 16
LEARNING_RATE = 4e-4
EPOCHS = 10
CLIP_GRAD_NORM = 5.0
TEACHER_FORCING_RATIO = 1.0

# OpenCV Visualization Settings
CV_FONT_SCALE = 0.65
CV_FONT_THICKNESS = 2
CV_BANNER_COLOR = (24, 24, 24)   # BGR Dark background banner
CV_TEXT_COLOR = (255, 255, 255)  # BGR White text
CV_ACCENT_COLOR = (0, 204, 255)  # BGR Gold/Yellow accent
