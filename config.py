import os
import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = "Aloevera leaf disease dataset"
OUTPUT_DIR = "outputs"
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "best_model.pt")
HISTORY_PATH = os.path.join(OUTPUT_DIR, "training_history.csv")
REPORT_PATH = os.path.join(OUTPUT_DIR, "test_classification_report.txt")
CONFUSION_MATRIX_PATH = os.path.join(OUTPUT_DIR, "confusion_matrix.png")


CLASS_NAMES = ["healthy", "rot", "rust"]
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15
SEED = 42


BACKBONE = "mobilenetv2_100"
NUM_CLASSES = len(CLASS_NAMES)
IMG_SIZE = 192
PRETRAINED = True


BATCH_SIZE = 16
NUM_WORKERS = 2
EPOCHS = 20
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 5
LR_PATIENCE = 2
USE_AMP = True


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
