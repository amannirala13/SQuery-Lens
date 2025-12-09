#!/bin/bash
# Run tests for SQL Category Classifier

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=============================================="
echo "SQL Category Classifier - Test & Validation"
echo "=============================================="
echo ""

# Activate virtual environment
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
else
    echo "Error: Virtual environment not found. Run ./setup.sh first."
    exit 1
fi

# Check for trained model
if [ ! -f "models/best_model.pt" ]; then
    echo "Error: No trained model found. Run ./run_train.sh first."
    exit 1
fi

# Parse arguments
MODE=${1:-test}

cd src/training

case $MODE in
    test)
        echo "Running test evaluation..."
        python test.py
        ;;
    validate)
        echo "Running validation..."
        python validate.py
        ;;
    both)
        echo "Running validation..."
        python validate.py
        echo ""
        echo "Running test evaluation..."
        python test.py
        ;;
    *)
        echo "Usage: ./run_test.sh [test|validate|both]"
        exit 1
        ;;
esac

echo ""
echo "Done! Results saved to logs/"
