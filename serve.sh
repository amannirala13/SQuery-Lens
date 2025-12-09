#!/bin/bash
# Start the API server for SQL Category Classifier

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=============================================="
echo "SQL Category Classifier - API Server"
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
PORT=${1:-8000}
HOST=${2:-0.0.0.0}

echo "Starting server at http://$HOST:$PORT"
echo "Press Ctrl+C to stop"
echo ""

# Run server
cd src/api
python -c "import uvicorn; uvicorn.run('api_server:app', host='$HOST', port=$PORT, reload=False)"
