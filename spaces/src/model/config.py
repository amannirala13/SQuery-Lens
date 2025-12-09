"""
Enhanced Configuration for SQL Query Analyzer
Multi-output model for query classification and table ranking
"""
import os
from pathlib import Path

# Paths - adjusted for new directory structure
BASE_DIR = Path(__file__).parent.parent.parent  # sql_classifier/
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

# Data files
DATA_PATH = DATA_DIR / "text-to-sql-100K.csv"
DATA_PATH_SMALL = DATA_DIR / "merged_data.csv"
SQL_CAT_PATH = DATA_DIR / "sql_cat.csv"

# Create directories
MODEL_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# Data split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.10
TEST_RATIO = 0.20

# Dataset size limit
MAX_SAMPLES = 10000  # Set to None for full dataset

# Model configuration
MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 256
BATCH_SIZE = 32
LEARNING_RATE = 2e-5
NUM_EPOCHS = 5
WARMUP_STEPS = 500
WEIGHT_DECAY = 0.01
EARLY_STOPPING_PATIENCE = 2

# Device configuration
import torch
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

RANDOM_SEED = 42

# Output files
BEST_MODEL_PATH = MODEL_DIR / "best_model_enhanced.pt"
LABEL_ENCODERS_PATH = MODEL_DIR / "label_encoders_enhanced.pkl"
TRAINING_HISTORY_PATH = LOGS_DIR / "training_history_enhanced.json"

# ============================================================
# LABEL DEFINITIONS
# ============================================================

# Complexity levels
COMPLEXITY_LABELS = ["simple", "medium", "complex"]

# SQL Keywords (multi-label)
SQL_KEYWORDS = [
    "SELECT", "INSERT", "UPDATE", "DELETE",
    "FROM", "WHERE", "JOIN", "INNER JOIN", "LEFT JOIN", "RIGHT JOIN",
    "GROUP BY", "HAVING", "ORDER BY", "LIMIT", "OFFSET",
    "DISTINCT", "COUNT", "SUM", "AVG", "MAX", "MIN",
    "AND", "OR", "IN", "NOT IN", "EXISTS", "NOT EXISTS",
    "BETWEEN", "LIKE", "IS NULL", "IS NOT NULL",
    "UNION", "INTERSECT", "EXCEPT",
    "SUBQUERY", "CASE", "WHEN", "THEN", "ELSE",
    "CREATE", "ALTER", "DROP", "TRUNCATE",
    "AS", "ON", "USING"
]

# Categories (single-label) - expanded
CATEGORIES = [
    "Data Manipulation",
    "Schema Definition", 
    "Set Operations",
    "Transaction Control",
    "Security",
    "Procedural SQL"
]

# Subcategories (multi-label)
SUBCATEGORIES = [
    "Read", "Write", "Aggregation", "Filtering", "Sorting",
    "Joining", "Grouping", "Limiting", "Subquery",
    "Windowing", "Set Logic", "Object DDL", "Constraints",
    "Indexing", "Transactions", "Permissions"
]

# Estimated table count
TABLE_COUNT_LABELS = ["1", "2", "3+"]
