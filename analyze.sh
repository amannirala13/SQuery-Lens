#!/bin/bash
# Quick inference script for SQL Query Analyzer
# Usage: ./classify.sh "your query here"

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check arguments
if [ -z "$1" ]; then
    echo "Usage: ./classify.sh \"your query here\""
    echo ""
    echo "Examples:"
    echo "  ./classify.sh \"Calculate total sales by region\""
    echo "  ./classify.sh \"Delete expired user sessions\""
    exit 1
fi

# Activate virtual environment
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
else
    echo "Error: Virtual environment not found. Run ./setup.sh first."
    exit 1
fi

# Check for trained model
if [ ! -f "models/best_model_enhanced.pt" ]; then
    echo "Error: No trained model found. Run ./run_train.sh first."
    exit 1
fi

QUERY="$1"

# Run inference
python3 -c "
import sys
sys.path.insert(0, 'src')

import warnings
warnings.filterwarnings('ignore')
import logging
logging.getLogger('transformers').setLevel(logging.ERROR)

import torch
from transformers import DistilBertTokenizer
from model import config
from model.data_processor import EnhancedDataProcessor
from model.model import EnhancedSQLAnalyzer

# Load model
model = EnhancedSQLAnalyzer()
checkpoint = torch.load('models/best_model_enhanced.pt', map_location='cpu', weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load encoders
processor = EnhancedDataProcessor()
processor.load_encoders()

tokenizer = DistilBertTokenizer.from_pretrained(config.MODEL_NAME)

# Predict
query = '''$QUERY'''

with torch.no_grad():
    encoding = tokenizer(
        query, add_special_tokens=True, max_length=config.MAX_LENGTH,
        padding='max_length', truncation=True, return_tensors='pt'
    )
    
    outputs = model(encoding['input_ids'], encoding['attention_mask'])
    
    # Complexity
    complexity_idx = torch.argmax(outputs['complexity'], dim=1).item()
    complexity_conf = torch.softmax(outputs['complexity'], dim=1)[0, complexity_idx].item()
    complexity = processor.complexity_encoder.inverse_transform([complexity_idx])[0]
    
    # Category
    category_idx = torch.argmax(outputs['category'], dim=1).item()
    category_conf = torch.softmax(outputs['category'], dim=1)[0, category_idx].item()
    category = processor.category_encoder.inverse_transform([category_idx])[0]
    
    # Table count
    table_count_idx = torch.argmax(outputs['table_count'], dim=1).item()
    table_count = processor.table_count_encoder.inverse_transform([table_count_idx])[0]
    
    # Keywords (multi-label)
    keyword_probs = torch.sigmoid(outputs['keywords'])[0]
    keywords = [config.SQL_KEYWORDS[i] for i in range(len(config.SQL_KEYWORDS)) if keyword_probs[i] > 0.5]
    
    # Subcategories (multi-label)
    subcat_probs = torch.sigmoid(outputs['subcategory'])[0]
    subcategories = [config.SUBCATEGORIES[i] for i in range(len(config.SUBCATEGORIES)) if subcat_probs[i] > 0.5]
    if not subcategories:
        max_idx = torch.argmax(subcat_probs).item()
        subcategories = [config.SUBCATEGORIES[max_idx]]

print(f'Query: {query}')
print()
print(f'Complexity:     {complexity} ({complexity_conf*100:.1f}%)')
print(f'Category:       {category} ({category_conf*100:.1f}%)')
print(f'Subcategories:  {subcategories}')
print(f'Keywords:       {keywords}')
print(f'Est. Tables:    {table_count}')
"
