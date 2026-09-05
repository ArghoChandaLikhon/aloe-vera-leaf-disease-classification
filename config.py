"""
Central configuration for the Aloe Vera disease classifier.

Change BACKBONE + IMG_SIZE + BATCH_SIZE here when moving from a laptop
baseline run to a bigger model on Kaggle/Colab. Nothing else in the
codebase needs to change.
"""

import os
import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = "Aloevera leaf disease dataset"  # must contain healthy/, rot/, rust/ subfolders
OUTPUT_DIR = "outputs"
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "best_model.pt")
HISTORY_PATH = os.path.join(OUTPUT_DIR, "training_history.csv")
REPORT_PATH = os.path.join(OUTPUT_DIR, "test_classification_report.txt")
CONFUSION_MATRIX_PATH = os.path.join(OUTPUT_DIR, "confusion_matrix.png")

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
# Alphabetical order -- this is also the order torchvision.ImageFolder
# assigns automatically from your folder names, so it must match exactly.
CLASS_NAMES = ["healthy", "rot", "rust"]
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15
SEED = 42

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
# Laptop baseline (2GB VRAM, e.g. MX250):  BACKBONE="mobilenetv2_100"  IMG_SIZE=192  BATCH_SIZE=16
# Kaggle/Colab full run (T4/P100):         BACKBONE="efficientnet_b3" IMG_SIZE=250  BATCH_SIZE=32
BACKBONE = "mobilenetv2_100"
NUM_CLASSES = len(CLASS_NAMES)
IMG_SIZE = 192
PRETRAINED = True

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
BATCH_SIZE = 16
NUM_WORKERS = 2  # keep low on a laptop CPU; raise to 4-8 on Kaggle
EPOCHS = 20
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 5
LR_PATIENCE = 2
USE_AMP = True  # mixed precision -- large memory saver on small GPUs, ignored on CPU

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
