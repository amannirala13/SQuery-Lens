#!/bin/bash
# Setup script for SQL Category Classifier
# Run this once to set up the environment

set -e  # Exit on error

echo "=============================================="
echo "SQL Category Classifier - Setup"
echo "=============================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check Python version
echo "Checking Python version..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
    echo -e "${GREEN}✓${NC} Python $PYTHON_VERSION found"
else
    echo "Error: Python 3 is required but not installed."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo ""
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo -e "${GREEN}✓${NC} Virtual environment created"
else
    echo -e "${GREEN}✓${NC} Virtual environment already exists"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate
echo -e "${GREEN}✓${NC} Virtual environment activated"

# Install Python dependencies
echo ""
echo "Installing Python dependencies..."
pip install --upgrade pip -q
pip install torch transformers pandas numpy scikit-learn tqdm fastapi uvicorn -q
pip install sentence-transformers accelerate datasets -q
echo -e "${GREEN}✓${NC} Python dependencies installed"

# Create necessary directories
echo ""
echo "Creating directories..."
mkdir -p data models logs
echo -e "${GREEN}✓${NC} Directories created"

# Check if model exists
if [ -f "models/best_model.pt" ]; then
    echo -e "${GREEN}✓${NC} Trained model found"
else
    echo -e "${YELLOW}!${NC} No trained model found. Run ./run_train.sh to train."
fi

# Check if data exists
if [ -f "data/text-to-sql-100K.csv" ]; then
    echo -e "${GREEN}✓${NC} Training data found (100K dataset)"
elif [ -f "data/merged_data.csv" ]; then
    echo -e "${GREEN}✓${NC} Training data found (small dataset)"
else
    echo -e "${YELLOW}!${NC} No training data found in data/ directory"
fi

# Install TypeScript dependencies
if [ -d "client" ]; then
    echo ""
    echo "Setting up TypeScript client..."
    cd client
    if command -v npm &> /dev/null; then
        npm install -q 2>/dev/null || true
        echo -e "${GREEN}✓${NC} TypeScript dependencies installed"
    else
        echo -e "${YELLOW}!${NC} npm not found. Install Node.js to use TypeScript client."
    fi
    cd ..
fi

echo ""
echo "=============================================="
echo -e "${GREEN}Setup Complete!${NC}"
echo "=============================================="
echo ""
echo "Next steps:"
echo "  1. Train classifier:     ./train.sh"
echo "  2. Train schema ranker:  ./train_ranker.sh"
echo "  3. Start API server:     ./serve.sh"
echo "  4. Analyze a query:      ./analyze.sh \"your query\""
echo "  5. Rank tables:          ./rank.sh \"query\" \"tables\""
echo ""
