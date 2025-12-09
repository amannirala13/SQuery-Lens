#!/bin/bash
# Training script for SQL Category Classifier

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=============================================="
echo "SQL Category Classifier - Training"
echo "=============================================="
echo ""

# Activate virtual environment
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
else
    echo "Error: Virtual environment not found. Run ./setup.sh first."
    exit 1
fi

# Check for training data
if [ ! -f "data/text-to-sql-100K.csv" ] && [ ! -f "data/merged_data.csv" ]; then
    echo "Error: No training data found in data/ directory."
    exit 1
fi

# Run training
cd src/training
python train.py

echo ""
echo "Training complete! Model saved to models/best_model.pt"
