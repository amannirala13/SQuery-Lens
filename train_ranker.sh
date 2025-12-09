#!/bin/bash
# Train Schema Ranker (Sentence-BERT for table/column relevance)
# Uses data_table_2 dataset with 2.8M samples

set -e
cd "$(dirname "${BASH_SOURCE[0]}")"

echo "=============================================="
echo "Schema Ranker Training (Sentence-BERT)"
echo "=============================================="

source venv/bin/activate

mkdir -p models/schema_ranker

cd src/schema_ranker
python train.py

echo ""
echo "Training complete!"
echo "Model saved to: models/schema_ranker/"
