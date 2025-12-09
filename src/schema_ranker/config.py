"""
Schema Ranker - Sentence-BERT based table/column relevance model
Trained on data_table_2 dataset (2.8M samples)
Works for BOTH table names AND column names (same semantic matching)
"""
from pathlib import Path
import torch

# Paths
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "data_table_2"
MODEL_DIR = BASE_DIR / "models" / "schema_ranker"
LOGS_DIR = BASE_DIR / "logs"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Data paths
TRAIN_PATH = DATA_DIR / "train.csv"
VAL_PATH = DATA_DIR / "validation.csv"
TEST_PATH = DATA_DIR / "test.csv"

# Model config
BASE_MODEL = "all-MiniLM-L6-v2"  # Fast, good quality
# Alternative: "all-mpnet-base-v2" (better but slower)

# Training config
MAX_SAMPLES = 100000  # Start with 100K for faster iteration, set to None for full
BATCH_SIZE = 64
NUM_EPOCHS = 3
LEARNING_RATE = 2e-5
WARMUP_RATIO = 0.1
RANDOM_SEED = 42

# Evaluation
EVAL_STEPS = 1000

# Model paths
BEST_MODEL_PATH = MODEL_DIR / "schema_ranker_model"
HISTORY_PATH = LOGS_DIR / "schema_ranker_history.json"

# Device
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"
